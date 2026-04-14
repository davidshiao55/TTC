"""Custom LLM wrapper that boots a V1 engine with FastTTS extensions.

``TTSLLM`` bypasses ``vllm.LLM.__init__`` so it can swap in
:class:`GeneratorLLMEngineV1` (for the generator, with Speculative Beam
Extension) or a plain V1 ``LLMEngine`` (for the verifier). After engine
creation, the attributes ``LLM.generate`` / ``LLM.encode`` rely on are
bootstrapped manually. ``score_outputs`` runs the Skywork PRM scoring
pipeline.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import cloudpickle
from tqdm.auto import tqdm

from vllm import LLM
from vllm.config import CompilationConfig, is_init_field
from vllm.config.model import ModelDType, TokenizerMode
from vllm.engine.arg_utils import EngineArgs, HfOverrides, PoolerConfig, RunnerOption
from vllm.inputs import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.outputs import PoolingRequestOutput
from vllm.pooling_params import PoolingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils.counter import Counter
from vllm.v1.engine.llm_engine import LLMEngine as VerifierLLMEngine

from .generator_engine_v1 import GeneratorLLMEngineV1
from .reward_utils import prepare_input, sigmoid

logger = init_logger(__name__)

SKYWORK_PRM_MODELS = (
    "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
    "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B",
)


def _build_compilation_config(
    compilation_config: Optional[Union[int, dict[str, Any], CompilationConfig]],
) -> CompilationConfig:
    """Normalize the caller-supplied compilation_config to a CompilationConfig."""
    if compilation_config is None:
        return CompilationConfig()
    if isinstance(compilation_config, int):
        return CompilationConfig(level=compilation_config)
    if isinstance(compilation_config, dict):
        valid = {k: v for k, v in compilation_config.items() if is_init_field(CompilationConfig, k)}
        return CompilationConfig(**valid)
    return compilation_config


def _resolve_worker_cls(kwargs: dict[str, Any]) -> None:
    """Cloudpickle a class-typed worker_cls in-place so it survives mp.spawn."""
    worker_cls = kwargs.get("worker_cls")
    if isinstance(worker_cls, type):
        kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)


class TTSLLM(LLM):
    def __init__(
        self,
        model: str,
        *,
        runner: RunnerOption = "auto",
        tokenizer: Optional[str] = None,
        tokenizer_mode: TokenizerMode = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        tensor_parallel_size: int = 1,
        dtype: ModelDType = "auto",
        quantization: Optional[QuantizationMethods] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        cpu_offload_gb: float = 0,
        enforce_eager: bool = False,
        disable_custom_all_reduce: bool = False,
        hf_token: Optional[Union[bool, str]] = None,
        hf_overrides: Optional[HfOverrides] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        pooler_config: Optional[PoolerConfig] = None,
        compilation_config: Optional[Union[int, dict[str, Any]]] = None,
        spec_beam_extension: bool = False,
        prefix_aware_scheduling: bool = False,
        **kwargs,
    ) -> None:
        """LLM constructor."""
        kwargs.setdefault("disable_log_stats", True)
        _resolve_worker_cls(kwargs)

        if hf_overrides is None:
            hf_overrides = {}

        # SBE requires priority scheduling — must be set at engine init time
        # (the scheduler queue type is chosen once in __init__).
        if spec_beam_extension:
            kwargs.setdefault("scheduling_policy", "priority")

        # Only pass seed when explicitly provided — thesis vLLM requires int.
        seed_kwargs = {"seed": seed} if seed is not None else {}

        engine_args = EngineArgs(
            model=model,
            runner=runner,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            allowed_local_media_path=allowed_local_media_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            **seed_kwargs,
            gpu_memory_utilization=gpu_memory_utilization,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            disable_custom_all_reduce=disable_custom_all_reduce,
            hf_token=hf_token,
            hf_overrides=hf_overrides,
            mm_processor_kwargs=mm_processor_kwargs,
            pooler_config=pooler_config,
            compilation_config=_build_compilation_config(compilation_config),
            **kwargs,
        )

        engine_cls = GeneratorLLMEngineV1 if runner in ("generate", "auto") else VerifierLLMEngine
        if engine_cls is GeneratorLLMEngineV1:
            logger.info(f"Using V1 engine with speculative beam extension: {spec_beam_extension}")
        logger.info(f"Prefix-aware scheduling enabled: {prefix_aware_scheduling}")

        self.llm_engine = engine_cls.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS,
        )
        if spec_beam_extension and isinstance(self.llm_engine, GeneratorLLMEngineV1):
            self.llm_engine.enable_spec_beam_extension()

        self.engine_class = type(self.llm_engine)
        self.request_counter = Counter()
        self.default_sampling_params: Union[dict[str, Any], None] = None
        self.model_path = model

        self._bootstrap_llm_attributes()

    def _bootstrap_llm_attributes(self) -> None:
        """Populate attributes that ``LLM.generate`` / ``LLM.encode`` expect."""
        self.model_config = self.llm_engine.model_config
        self.renderer = getattr(self.llm_engine, "renderer", None)
        self.runner_type = self.model_config.runner_type
        self.supported_tasks = self.llm_engine.get_supported_tasks()
        self.pooling_io_processors = {}
        self.io_processor = getattr(self.llm_engine, "io_processor", None)
        self.input_processor = getattr(self.llm_engine, "input_processor", None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_text(self, prompts, **kwargs):
        assert isinstance(self.llm_engine, GeneratorLLMEngineV1), (
            "GeneratorLLMEngineV1 is required for generate_text"
        )
        return self.generate(prompts, **kwargs)

    def encode(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        pooling_params: Optional[Union[PoolingParams, Sequence[PoolingParams]]] = None,
        *,
        use_tqdm: Union[bool, Callable[..., tqdm]] = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        priority: Optional[list[int]] = None,
        **_kwargs,
    ) -> list[PoolingRequestOutput]:
        if pooling_params is None:
            pooling_params = PoolingParams()
        return self._run_completion(
            prompts=prompts,
            params=pooling_params,
            output_type=PoolingRequestOutput,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            priority=priority,
        )

    def score_outputs(self, questions, outputs, system_prompt, **kwargs):
        if self.model_path not in SKYWORK_PRM_MODELS:
            raise NotImplementedError(f"score_outputs not implemented for model_path: {self.model_path}")
        return self._score_outputs_skywork(questions, outputs, system_prompt, **kwargs)

    # ------------------------------------------------------------------
    # Skywork PRM scoring (decomposed into 3 phases)
    # ------------------------------------------------------------------

    def _score_outputs_skywork(self, questions, outputs, _, *,
                               skip_reading_prefix_cache: bool = False, **kwargs):
        """Pure PRM scoring — returns raw per-step scores.

        When ``skip_reading_prefix_cache=False`` (default), within-batch shared
        prefixes hit the KV cache and cached step boundaries come back as
        ``None``; callers must fill them via the search-layer step-hash
        propagation (see ``_score_and_assign``). Single-shot callers that
        don't run propagation (e.g. ``best_of_n``) pass ``True`` to get a
        None-free scoring at the cost of re-encoding the shared prefix.
        """
        prompts, flat_reward_flags, boundaries = self._build_skywork_scoring_prompts(
            questions, outputs,
        )
        rewards = self.encode(
            prompts,
            pooling_params=PoolingParams(
                skip_reading_prefix_cache=skip_reading_prefix_cache,
            ),
        )
        flat_step_rewards = _extract_step_rewards(rewards, flat_reward_flags)
        return _rebuild_nested_scores(flat_step_rewards, boundaries)

    def _build_skywork_scoring_prompts(self, questions, outputs):
        tokenizer = self.get_tokenizer()
        max_model_len = self.model_config.max_model_len
        flat_input_ids = []
        flat_reward_flags = []
        boundaries = []
        for question, output in zip(questions, outputs):
            processed = [
                prepare_input(question, o, tokenizer=tokenizer, max_model_len=max_model_len)
                for o in output
            ]
            for input_ids, _steps, reward_flags in processed:
                flat_input_ids.append(input_ids)
                flat_reward_flags.append(reward_flags)
            boundaries.append(len(output))
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in flat_input_ids]
        return prompts, flat_reward_flags, boundaries


def _extract_step_rewards(rewards, flat_reward_flags):
    """Pull per-step scores out of PRM reward embeddings.

    With prefix caching, cached prefix tokens produce no hidden states — those
    step boundaries get ``None`` placeholders that are filled later in the
    search layer via step-hash propagation.
    """
    flat_step_rewards = []
    for idx, reward in enumerate(rewards):
        reward_flag = flat_reward_flags[idx]
        reward_embedding = reward.outputs.data.tolist()
        offset = len(reward_flag) - len(reward_embedding)
        if offset < 0:
            logger.error(
                f"reward_embedding ({len(reward_embedding)}) larger than "
                f"reward_flag ({len(reward_flag)}) — unexpected. Prompt idx={idx}"
            )
        step_reward = []
        for i, flag in enumerate(reward_flag):
            if flag != 1:
                continue
            local_idx = i - offset
            if local_idx < 0:
                step_reward.append(None)  # cached — fill in beam_search
            elif local_idx >= len(reward_embedding):
                break
            else:
                step_reward.append(sigmoid(reward_embedding[local_idx][0]))
        flat_step_rewards.append(step_reward)
    return flat_step_rewards


def _rebuild_nested_scores(flat_step_rewards, boundaries):
    """Restore the per-question / per-output nesting from a flat rewards list."""
    all_scores = []
    idx = 0
    for num_outputs in boundaries:
        all_scores.append(flat_step_rewards[idx:idx + num_outputs])
        idx += num_outputs
    return all_scores
