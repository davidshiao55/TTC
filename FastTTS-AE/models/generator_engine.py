# Custom synchronous LLM engine
from vllm.engine.llm_engine import LLMEngine, SchedulerOutputState, SchedulerContext
import copy
import time
from collections import Counter as collectionsCounter
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import (TYPE_CHECKING, Any, Callable, ClassVar, Deque, Dict,
                    Iterable, List, Literal, Mapping, NamedTuple, Optional)
from typing import Sequence as GenericSequence
from typing import Set, Type, Union, cast

import torch
from typing_extensions import TypeVar

import vllm.envs as envs
from vllm.config import (DecodingConfig, LoRAConfig, ModelConfig,
                         ObservabilityConfig, ParallelConfig, SchedulerConfig,
                         VllmConfig)
from vllm.core.scheduler import ScheduledSequenceGroup, SchedulerOutputs
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics_types import StatLoggerBase, Stats
from vllm.engine.output_processor.interfaces import (
    SequenceGroupOutputProcessor)
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.engine.output_processor.util import create_output_by_sequence_group
from vllm.entrypoints.openai.logits_processors import (
    get_logits_processors as get_openai_logits_processors)
from vllm.executor.executor_base import ExecutorBase
from vllm.inputs import ProcessorInputs, PromptType, SingletonInputs
from vllm.inputs.parse import split_enc_dec_inputs
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.logits_process import get_bad_words_logits_processors
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding import (
    get_local_guided_decoding_logits_processor)
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.processing import EncDecMultiModalProcessor
from vllm.outputs import (PoolingRequestOutput, RequestOutput,
                          RequestOutputFactory)
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.sequence import (ExecuteModelRequest, ParallelSampleSequenceGroup,
                           PoolingSequenceGroupOutput, Sequence, SequenceGroup,
                           SequenceGroupBase, SequenceGroupMetadata,
                           SequenceGroupOutput, SequenceStatus)
from vllm.tracing import (SpanAttributes, SpanKind, extract_trace_context,
                          init_tracer)
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import (
    TokenizerGroup, init_tokenizer_from_configs)
from vllm.usage.usage_lib import (UsageContext, is_usage_stats_enabled,
                                  usage_message)
from vllm.utils import Counter, Device, resolve_obj_by_qualname, weak_bind
from vllm.version import __version__ as VLLM_VERSION
from vllm.worker.model_runner_base import InputProcessingError
from vllm.sequence import SequenceStatus
from vllm.engine.llm_engine import SchedulerContext

from .custom_scheduler import CustomScheduler
from .spec_stopchecker import SpecStopChecker, is_finished_stopped_with_stop
from vllm.engine.output_processor.single_step import SingleStepOutputProcessor
from .numbers import SPEC_BEAM_CANDIDATE_PRIORITY

logger = init_logger(__name__)

class GeneratorLLMEngine(LLMEngine):
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[ExecutorBase],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
    ) -> None:
        logger.info(
            f"Using GeneratorLLMEngine with vLLM version {VLLM_VERSION}"
        )
        vllm_config.scheduler_config.scheduler_cls = CustomScheduler
        super().__init__(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            mm_registry=mm_registry,
            use_cached_outputs=use_cached_outputs,
        )
        self.spec_beam_extension_enabled = False
    
    def enable_spec_beam_extension(self) -> bool:
        """Set up speculative beam extension."""
        assert self.scheduler_config.num_scheduler_steps == 1, (
            "Speculative beam extension is only supported for single-step "
            "scheduling for now"
        )
        # Set scheduler policy to priority for spec beam extension
        self.scheduler_config.policy = "priority"
        if not self.model_config.skip_tokenizer_init:
            self.tokenizer = self._init_tokenizer()
            self.detokenizer = Detokenizer(self.tokenizer)
            tokenizer_group = self.get_tokenizer_group()
        else:
            self.tokenizer = None
            self.detokenizer = None
            tokenizer_group = None
        # Ensure that the function doesn't contain a reference to self,
        # to avoid engine GC issues
        def get_tokenizer_for_seq(sequence: Sequence) -> AnyTokenizer:
            assert tokenizer_group, ("tokenizer_group cannot be None, "
                                    "make sure skip_tokenizer_init is False")
            return tokenizer_group.get_lora_tokenizer(sequence.lora_request)
        self.output_processor = SingleStepOutputProcessor(
            self.scheduler_config,
            self.detokenizer,
            self.scheduler,
            self.seq_counter,
            stop_checker=SpecStopChecker(self.scheduler_config.max_model_len,
                                        get_tokenizer_for_seq,
                                        self.scheduler[0]),)
        self.spec_beam_extension_enabled = True
    
    
    def _process_model_outputs_spec(self,
                                    ctx: SchedulerContext,
                                    request_id: Optional[str] = None) -> None:
        now = time.time()

        if len(ctx.output_queue) == 0:
            return None

        # Get pending async postprocessor
        if request_id:
            # When we process only one request, no pop is required
            # (since later we will process all of the rest)
            (outputs, seq_group_metadata_list, scheduler_outputs, is_async,
             is_last_step, is_first_step_output, skip) = ctx.output_queue[0]
        else:
            (outputs, seq_group_metadata_list, scheduler_outputs, is_async,
             is_last_step, is_first_step_output,
             skip) = ctx.output_queue.popleft()

        # Sanity check
        assert len(seq_group_metadata_list) == len(
            scheduler_outputs.scheduled_seq_groups)

        has_multiple_outputs: bool = len(outputs) > 1
        outputs_by_sequence_group: List[List[SequenceGroupOutput]]
        if has_multiple_outputs:
            raise NotImplementedError(
                "Multiple outputs per step are not supported in speculative "
                "beam extension. Please use single-step scheduling.")
        else:
            outputs_by_sequence_group = outputs

        # Determine the requests we need to operate on
        if request_id:
            indices = []
            for i, seq_group_meta in enumerate(seq_group_metadata_list):
                if seq_group_meta.request_id == request_id:
                    assert i not in skip  # Cannot be called twice
                    indices.append(i)
                    break

            # If the request_id was not found, then it means that
            # this is a new request that has no pending async
            # postprocessor
            if not indices:
                return
        else:
            indices = range(len(seq_group_metadata_list))  # type: ignore

        finished_before: List[int] = []
        finished_now: List[int] = []
        running_now: List[int] = []
        spec_beam_extension_now: List[int] = []
        all_finished = True
        for i in indices:
            if i in skip:
                continue

            seq_group_meta = seq_group_metadata_list[i]
            scheduled_seq_group = scheduler_outputs.scheduled_seq_groups[i]

            seq_group: SequenceGroup = scheduled_seq_group.seq_group

            if seq_group.is_finished():
                finished_before.append(i)
                continue

            output: List[SequenceGroupOutput] = [outputs_by_sequence_group[0][i]]

            if not is_async:
                if self.scheduler_config.is_multi_step:
                    # Updates happen only if the sequence is prefill
                    self._update_num_computed_tokens_for_multi_step_prefill(
                        seq_group, seq_group_meta, is_first_step_output)
                else:
                    seq_group.update_num_computed_tokens(
                        seq_group_meta.token_chunk_size or 0)

            if outputs:
                for o in outputs:
                    if (isinstance(o, SamplerOutput)
                            and seq_group.metrics is not None):
                        if seq_group.metrics.model_forward_time is not None:
                            seq_group.metrics.model_forward_time += (
                                o.model_forward_time or 0)
                        else:
                            seq_group.metrics.model_forward_time = (
                                o.model_forward_time)
                        if seq_group.metrics.model_execute_time is not None:
                            seq_group.metrics.model_execute_time += (
                                o.model_execute_time or 0)
                        else:
                            seq_group.metrics.model_execute_time = (
                                o.model_execute_time)

            if self.model_config.runner_type == "pooling":
                raise NotImplementedError(
                    "Pooling is not supported in speculative beam extension. "
                    "Please use single-step scheduling.")
            else:
                assert seq_group.first_seq.status == SequenceStatus.RUNNING, f"Sequence {seq_group.first_seq.seq_id} status: {SequenceStatus.get_finished_reason(seq_group.first_seq.status)}"
                self.output_processor.process_prompt_logprob(seq_group, output)
                if seq_group_meta.do_sample:
                    self.output_processor.process_outputs(
                        seq_group, output, is_async)     
                    if is_finished_stopped_with_stop(seq_group.first_seq):
                        seq_group.priority = SPEC_BEAM_CANDIDATE_PRIORITY
                all_finished &= is_finished_stopped_with_stop(seq_group.first_seq) or seq_group.first_seq.is_finished()
                # if is_finished_stopped_with_stop(seq_group.first_seq) or seq_group.first_seq.is_finished():
                #     logger.info(
                #         f"Sequence {seq_group.first_seq.seq_id}; status: {SequenceStatus.get_finished_reason(seq_group.first_seq.status)}; "
                #         f"Stop Reason: {repr(seq_group.first_seq.stop_reason)};"
                #         f"Priority: {seq_group.priority};"
                #     ) 

            if seq_group.is_finished():
                finished_now.append(i)
            if is_finished_stopped_with_stop(seq_group.first_seq):
                spec_beam_extension_now.append(i)
            if not is_finished_stopped_with_stop(seq_group.first_seq) and not seq_group.first_seq.is_finished():
                running_now.append(seq_group.first_seq.seq_id)
        # logger.info(f"Spec Beam Extension Now: {len(spec_beam_extension_now)}")
        # logger.info(f"Running Now: {len(running_now)}")
        while len(spec_beam_extension_now) + len(running_now) > 256 and len(spec_beam_extension_now) > 0:
            # logger.info(f"Spec Beam Extension Now: {len(spec_beam_extension_now)}")
            # logger.info(f"Running Now: {len(running_now)}")
            i = spec_beam_extension_now.pop(0)
            seq_group = scheduler_outputs.scheduled_seq_groups[i].seq_group
            seq_group.first_seq.status = SequenceStatus.FINISHED_STOPPED
            for scheduler in self.scheduler:
                scheduler.free_seq(seq_group.first_seq)
            finished_now.append(i)
        
        if all_finished:
            for i in indices: 
                if i in skip or i in finished_before or i in finished_now:
                    continue
                scheduled_seq_group = scheduler_outputs.scheduled_seq_groups[i]
                seq = scheduled_seq_group.seq_group.first_seq
                assert is_finished_stopped_with_stop(seq), f"Sequence {seq.seq_id} status: {seq.status}, stop reason: {seq.stop_reason}"
                seq.status = SequenceStatus.FINISHED_STOPPED
                for scheduler in self.scheduler:
                    scheduler.free_seq(seq)
                finished_now.append(i)
        
        # Generate outputs for the requests that finished this iteration
        for i in finished_now:
            scheduled_seq_group = scheduler_outputs.scheduled_seq_groups[i]

            seq_group = scheduled_seq_group.seq_group
            seq_group.maybe_set_first_token_time(now)
            if not seq_group.is_prefill():
                seq_group.set_last_token_time(now)
            request_output = RequestOutputFactory.create(
                seq_group,
                self.seq_id_to_seq_group,
                use_cache=self.use_cached_outputs)
            if request_output:
                ctx.request_outputs.append(request_output)

        # When we process a single request, we skip it for the next time,
        # and invoke the request output callback (if there was final output)
        if request_id:
            assert len(indices) == 1
            skip.append(indices[0])

            if (finished_now
                    and self.process_request_outputs_callback is not None):
                self.process_request_outputs_callback(ctx.request_outputs)
                ctx.request_outputs.clear()
            return

        # Free currently finished requests
        if finished_now:
            for scheduler in self.scheduler:
                scheduler.free_finished_seq_groups()

        # For multi-step without streaming, don't create outputs each iteration
        if not is_last_step and not ctx.multi_step_stream_outputs:
            # Immediately process request outputs here (if callback is given)
            if (finished_now
                    and self.process_request_outputs_callback is not None):
                self.process_request_outputs_callback(ctx.request_outputs)
                ctx.request_outputs.clear()
            return

        # Create the outputs
        for i in indices:
            if i in skip or i in finished_before or i in finished_now:
                continue  # Avoids double processing

            scheduled_seq_group = scheduler_outputs.scheduled_seq_groups[i]

            seq_group = scheduled_seq_group.seq_group
            seq_group.maybe_set_first_token_time(now)
            if not seq_group.is_prefill():
                seq_group.set_last_token_time(now)
            request_output = RequestOutputFactory.create(
                seq_group,
                self.seq_id_to_seq_group,
                use_cache=self.use_cached_outputs)
            if request_output:
                ctx.request_outputs.append(request_output)

        # For multi-step with streaming, create outputs each iteration
        if not is_last_step and ctx.multi_step_stream_outputs:
            # Immediately process request outputs here (if callback is given)
            if self.process_request_outputs_callback is not None:
                self.process_request_outputs_callback(ctx.request_outputs)
                ctx.request_outputs.clear()
            return

        for seq_group in scheduler_outputs.ignored_seq_groups:
            params = seq_group.sampling_params
            if params is not None and params.output_kind == (
                    RequestOutputKind.DELTA) and not seq_group.is_finished():
                continue

            request_output = RequestOutputFactory.create(
                seq_group,
                self.seq_id_to_seq_group,
                use_cache=self.use_cached_outputs,
            )
            if request_output:
                ctx.request_outputs.append(request_output)

        # Immediately process request outputs here (if callback is given)
        if (ctx.request_outputs
                and self.process_request_outputs_callback is not None):
            self.process_request_outputs_callback(ctx.request_outputs)
            ctx.request_outputs.clear()

        # For async case, we need to record the stats here.
        # For non-async case, the stats are done in the
        # LLMEngine/AsyncLLMEngine directly
        if is_async:
            # Log stats.
            self.do_log_stats(scheduler_outputs, outputs, finished_before,
                              skip)

            # Tracing
            self.do_tracing(scheduler_outputs, finished_before)

        return None
    
    def _process_model_outputs(self,
                               ctx: SchedulerContext,
                               request_id: Optional[str] = None) -> None:
        """Apply the model output to the sequences in the scheduled seq groups
        and return responses.

        ctx: The virtual engine context to work on
        request_id: If provided, then only this request is going to be processed
        """
        # inspect the scheduler outputs
        # if len(ctx.output_queue) > 0:
        #     (_, _, scheduler_outputs, _,
        #         _, _, _) = ctx.output_queue[0]
        #     logger.info(f"Number of prefill groups: {scheduler_outputs.num_prefill_groups}")
        #     logger.info(f"Number of running groups: {len(self.scheduler[0].running)}")
        #     logger.info(f"Number of waiting groups: {len(self.scheduler[0].waiting)}")
            # for seq_group in scheduler_outputs.scheduled_seq_groups:
            #     seq = seq_group.seq_group.first_seq
            #     logger.info(f"Sequence {seq.seq_id}; status: {SequenceStatus.get_finished_reason(seq.status)}; Is Prefill: {seq.is_prefill()};")
        if len(ctx.output_queue) > 0 and self.spec_beam_extension_enabled:
            (_, _, scheduler_outputs, _,
                _, _, _) = ctx.output_queue[0]
            if len(scheduler_outputs.scheduled_seq_groups) > 0:
                sampling_params = scheduler_outputs.scheduled_seq_groups[0].seq_group.sampling_params
                if sampling_params.stop:
                    return self._process_model_outputs_spec(ctx, request_id)
        return super()._process_model_outputs(ctx, request_id)
        
        