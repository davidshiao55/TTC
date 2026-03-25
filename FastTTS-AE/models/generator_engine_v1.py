# V1 Engine Support for FastTTS
from vllm.engine.llm_engine import SchedulerOutputState, SchedulerContext
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

# V1 Engine Support
from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
from vllm.v1.executor.abstract import Executor
from vllm.v1.metrics.loggers import StatLoggerFactory
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.processor import Processor
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.metrics.stats import IterationStats
from vllm.v1.metrics.reader import get_metrics_snapshot
from vllm.v1.engine.parallel_sampling import ParentRequest

class GeneratorLLMEngineV1(V1LLMEngine):
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        multiprocess_mode: bool = False,
    ) -> None:
        logger.info(
            f"Using GeneratorLLMEngineV1 with vLLM version {VLLM_VERSION}"
        )
        # Note: V1 engine doesn't use scheduler_cls in the same way as V0
        # The V1 engine uses EngineCoreClient which handles scheduling differently
        super().__init__(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            mm_registry=mm_registry,
            use_cached_outputs=use_cached_outputs,
            multiprocess_mode=multiprocess_mode,
        )
        self.spec_beam_extension_enabled = False
    
    def enable_spec_beam_extension(self) -> bool:
        """Set up speculative beam extension for V1 engine."""
        # V1 engine uses a different architecture, so we need to adapt
        # the speculative beam extension for the V1 engine's components
        
        # For V1, we need to check if we can access the tokenizer
        if not self.model_config.skip_tokenizer_init:
            tokenizer_group = self.get_tokenizer_group()
        else:
            tokenizer_group = None
            
        # Ensure that the function doesn't contain a reference to self,
        # to avoid engine GC issues
        def get_tokenizer_for_seq(sequence: Sequence) -> AnyTokenizer:
            assert tokenizer_group, ("tokenizer_group cannot be None, "
                                    "make sure skip_tokenizer_init is False")
            return tokenizer_group.get_lora_tokenizer(sequence.lora_request)
        
        # V1 engine uses OutputProcessor instead of the V0's output processor
        # We need to create a custom output processor that supports speculative beam extension
        # For now, we'll set a flag and handle it in the step method
        self.spec_beam_extension_enabled = True
        logger.info("Speculative beam extension enabled for V1 engine")
        
    def step(self) -> Union[list[RequestOutput], list[PoolingRequestOutput]]:
        """Override step method to support speculative beam extension in V1."""
        if self.should_execute_dummy_batch:
            self.should_execute_dummy_batch = False
            self.engine_core.execute_dummy_batch()
            return []

        # 1) Get EngineCoreOutput from the EngineCore.
        outputs = self.engine_core.get_output()

        # 2) Process EngineCoreOutputs with speculative beam extension if enabled
        iteration_stats = IterationStats() if self.log_stats else None
        
        if self.spec_beam_extension_enabled:
            # Apply speculative beam extension logic here
            # This is a simplified version - you may need to adapt this
            # based on the specific V1 engine architecture
            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,
                engine_core_timestamp=outputs.timestamp,
                iteration_stats=iteration_stats)
            
            # Apply speculative beam extension logic
            # Note: This is a placeholder - the actual implementation
            # would need to be adapted for V1's different architecture
            if hasattr(self, '_apply_spec_beam_extension_v1'):
                processed_outputs = self._apply_spec_beam_extension_v1(
                    processed_outputs, outputs)
        else:
            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,
                engine_core_timestamp=outputs.timestamp,
                iteration_stats=iteration_stats)

        # 3) Abort any reqs that finished due to stop strings.
        self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        # 4) Record stats
        if self.stat_logger is not None:
            assert outputs.scheduler_stats is not None
            self.stat_logger.record(scheduler_stats=outputs.scheduler_stats,
                                    iteration_stats=iteration_stats)

        return processed_outputs.request_outputs
    
    def _apply_spec_beam_extension_v1(self, processed_outputs, outputs):
        """Apply speculative beam extension logic for V1 engine."""
        # This is a placeholder implementation
        # The actual implementation would need to be adapted for V1's architecture
        # which uses EngineCoreClient, Processor, and OutputProcessor instead of
        # the direct scheduler approach of V0
        
        # For now, we'll just return the processed outputs as-is
        # You would need to implement the speculative beam extension logic
        # based on the V1 engine's specific architecture
        return processed_outputs
