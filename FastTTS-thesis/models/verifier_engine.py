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

logger = init_logger(__name__)

class VerifierLLMEngine(LLMEngine):
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
            f"Using VerifierLLMEngine with vLLM version {VLLM_VERSION}"
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
        