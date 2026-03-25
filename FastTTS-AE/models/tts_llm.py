from vllm import LLM, SamplingParams
from collections.abc import Sequence
from typing import (TYPE_CHECKING, Any, Callable, ClassVar, Optional, Union,
                    cast, overload)
import torch
from collections import deque

import cloudpickle
from tqdm.auto import tqdm

from vllm.config import (CompilationConfig, ModelDType, TokenizerMode,
                         is_init_field)
from vllm.engine.arg_utils import (EngineArgs, HfOverrides, PoolerConfig,
                                   TaskOption)
from vllm.engine.llm_engine import LLMEngine
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         ChatTemplateContentFormatOption,
                                         apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         parse_chat_messages,
                                         resolve_chat_template_content_format)
from vllm.entrypoints.score_utils import (_cosine_similarity,
                                          _validate_score_input_lens)
from vllm.entrypoints.utils import _validate_truncation_size
from vllm.inputs import PromptType, SingletonPrompt, TextPrompt, TokensPrompt
from vllm.inputs.parse import parse_and_batch_prompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding.guided_fields import (
    GuidedDecodingRequest, LLMGuidedOptions)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.outputs import (ClassificationRequestOutput, EmbeddingRequestOutput,
                          PoolingRequestOutput, RequestOutput,
                          ScoringRequestOutput)
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import (GuidedDecodingParams, RequestOutputKind, SamplingParams)
from vllm.transformers_utils.tokenizer import (AnyTokenizer, MistralTokenizer, get_cached_tokenizer)
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter, Device, deprecate_kwargs, is_list_of

from .generator_engine import GeneratorLLMEngine
from .generator_engine_v1 import GeneratorLLMEngineV1
from .verifier_engine import VerifierLLMEngine

from .reward_utils import prepare_input, sigmoid

import torch.cuda.nvtx as nvtx

logger = init_logger(__name__)

# Custom synchronous LLM wrapper
class TTSLLM(LLM):
    def __init__(
        self,
        model: str,
        *,
        task: TaskOption = "auto",
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
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: bool = False,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        disable_async_output_proc: bool = False,
        hf_token: Optional[Union[bool, str]] = None,
        hf_overrides: Optional[HfOverrides] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        override_pooler_config: Optional[PoolerConfig] = None,
        compilation_config: Optional[Union[int, dict[str, Any]]] = None,
        # New parameters
        spec_beam_extension: bool = False,
        prefix_aware_scheduling: bool = False,
        **kwargs,
    ) -> None:
        """LLM constructor."""

        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            # if the worker_cls is not qualified string name,
            # we serialize it using cloudpickle to avoid pickling issues
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        if hf_overrides is None:
            hf_overrides = {}

        if compilation_config is not None:
            if isinstance(compilation_config, int):
                compilation_config_instance = CompilationConfig(
                    level=compilation_config)
            elif isinstance(compilation_config, dict):
                predicate = lambda x: is_init_field(CompilationConfig, x[0])
                compilation_config_instance = CompilationConfig(
                    **dict(filter(predicate, compilation_config.items())))
            else:
                compilation_config_instance = compilation_config
        else:
            compilation_config_instance = CompilationConfig()

        engine_args = EngineArgs(
            model=model,
            task=task,
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
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            disable_async_output_proc=disable_async_output_proc,
            hf_token=hf_token,
            hf_overrides=hf_overrides,
            mm_processor_kwargs=mm_processor_kwargs,
            override_pooler_config=override_pooler_config,
            compilation_config=compilation_config_instance,
            **kwargs,
        )

        # Create the Engine (autoselects V0 vs V1)
        if task == 'generate' or task == 'auto': 
            # Check if we should use V1 engine
            import vllm.envs as envs
            if envs.VLLM_USE_V1 and GeneratorLLMEngineV1 is not None:
                engine_cls = GeneratorLLMEngineV1
                logger.info(f"Using V1 engine with speculative beam extension: {spec_beam_extension}")
            else:
                engine_cls = GeneratorLLMEngine
                logger.info(f"Using V0 engine with speculative beam extension: {spec_beam_extension}")
        else:
            engine_cls = VerifierLLMEngine
        logger.info(f"Prefix-aware scheduling enabled: {prefix_aware_scheduling}")
        self.llm_engine = engine_cls.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS)
        if spec_beam_extension and isinstance(self.llm_engine, (GeneratorLLMEngine, GeneratorLLMEngineV1)):
            self.llm_engine.enable_spec_beam_extension()
        # if prefix_aware_scheduling:
        #     [scheduler.enable_prefix_aware_scheduling() for scheduler in self.llm_engine.scheduler]
        self.engine_class = type(self.llm_engine)

        self.request_counter = Counter()
        self.default_sampling_params: Union[dict[str, Any], None] = None
        self.model_path = model  # Store the model path for score_outputs dispatch

    def generate_text(self, prompts, **kwargs):
        assert isinstance(self.llm_engine, (GeneratorLLMEngine, GeneratorLLMEngineV1)), "GeneratorLLMEngine or GeneratorLLMEngineV1 is required for generate_text"
        request_outputs = self.generate(prompts, **kwargs)
        return request_outputs

    def _validate_and_add_requests(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                      Sequence[PoolingParams]],
        *,
        use_tqdm: Union[bool, Callable[..., tqdm]] = True,
        lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        guided_options: Optional[GuidedDecodingRequest] = None,
        priority: Optional[list[int]] = None,
    ) -> None:
        if priority is not None:
            # For V1 engine, priority handling might be different
            if hasattr(self.llm_engine, 'scheduler'):
                for scheduler in self.llm_engine.scheduler:
                    scheduler.scheduler_config.policy = "priority"
        super()._validate_and_add_requests(
            prompts=prompts,
            params=params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            guided_options=guided_options,
            tokenization_kwargs=tokenization_kwargs,
            priority=priority)
        if priority is not None and hasattr(self.llm_engine, 'scheduler'):
            for scheduler in self.llm_engine.scheduler:
                # sort the requests by priority
                requests = scheduler.waiting
                scheduler.waiting = deque(sorted(requests, key=lambda x: x.priority))
    
    def encode(
        self,
        prompts: Union[Union[PromptType, Sequence[PromptType]],
                       Optional[Union[str, list[str]]]] = None,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        prompt_token_ids: Optional[Union[list[int], list[list[int]]]] = None,
        truncate_prompt_tokens: Optional[int] = None,
        use_tqdm: Union[bool, Callable[..., tqdm]] = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: Optional[list[int]] = None,
    ) -> list[PoolingRequestOutput]:
        runner_type = self.llm_engine.model_config.runner_type
        if runner_type != "pooling":
            messages = ["LLM.encode() is only supported for pooling models."]

            supported_runner_types = self.llm_engine.model_config \
                .supported_runner_types
            if "pooling" in supported_runner_types:
                messages.append(
                    "Your model supports the 'pooling' runner, but is "
                    f"currently initialized for the '{runner_type}' runner. "
                    "Please initialize vLLM using `--task embed`, "
                    "`--task classify`, `--task score` etc.")

            raise ValueError(" ".join(messages))

        if prompt_token_ids is not None:
            parsed_prompts = self._convert_v1_inputs(
                prompts=cast(Optional[Union[str, list[str]]], prompts),
                prompt_token_ids=prompt_token_ids,
            )
        else:
            parsed_prompts = cast(Union[PromptType, Sequence[PromptType]],
                                  prompts)

        if pooling_params is None:
            # Use default pooling params.
            pooling_params = PoolingParams()
        elif isinstance(pooling_params, PoolingParams):
            pooling_params.verify(self.llm_engine.model_config)
        else:
            for pooling_param in pooling_params:
                pooling_param.verify(self.llm_engine.model_config)

        tokenization_kwargs: dict[str, Any] = {}
        _validate_truncation_size(self.llm_engine.model_config.max_model_len,
                                  truncate_prompt_tokens, tokenization_kwargs)

        self._validate_and_add_requests(
            prompts=parsed_prompts,
            params=pooling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return self.engine_class.validate_outputs(outputs,
                                                  PoolingRequestOutput)


    def score_outputs(self, questions, outputs, system_prompt, **kwargs):
        """Dispatch to the correct scoring implementation based on model_path."""
        # assert isinstance(self.llm_engine, VerifierLLMEngine), "VerifierLLMEngine is required for score_outputs"
        if self.model_path == "peiyi9979/math-shepherd-mistral-7b-prm":
            return self._score_outputs_math_shepherd(questions, outputs, system_prompt, **kwargs)
        elif self.model_path in [
            "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
            "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B"
        ]:
            return self._score_outputs_skywork(questions, outputs, system_prompt, **kwargs)
        else:
            raise NotImplementedError(f"score_outputs not implemented for model_path: {self.model_path}")

    def _score_outputs_math_shepherd(self, questions, outputs, system_prompt, **kwargs):
        inputs_for_prm = []
        lengths = []
        tokenizer = self.get_tokenizer()
        for question, output in zip(questions, outputs):
            prompt = system_prompt + "\n" + question + "\n"
            special_outputs = [o.replace("\n\n", " ки\n\n") for o in output]
            special_outputs = [
                o + " ки" if o[-2:] != "\n\n" else o for o in special_outputs
            ]
            special_outputs = [f"{prompt} {o}" for o in special_outputs]
            special_outputs = [TokensPrompt(prompt_token_ids=tokenizer.encode(o, truncation=True, max_length=4096)) for o in special_outputs]
            inputs_for_prm.extend(special_outputs)
            lengths.append(len(output))
        nvtx.range_push("encode")
        request_outputs = self.encode(inputs_for_prm, **kwargs)
        nvtx.range_pop()
        output_scores = [[output.outputs.data[:, 0].tolist()] for output in request_outputs] if len(outputs) > 0 else []
        # cumulative_lengths = list(accumulate(lengths))
        # output_scores = [
        #     output_scores[i:j].tolist()
        #     for i, j in zip([0] + cumulative_lengths[:-1], cumulative_lengths)
        # ]
        # for output_score, output in zip(output_scores, outputs):
        #     assert len(output_score) == len(output), f"{len(output_score)} != {len(output)}"
        return output_scores

    def _score_outputs_skywork(self, questions, outputs, _, **kwargs):
        all_scores = []
        tokenizer = self.get_tokenizer()
        flat_input_ids = []
        flat_reward_flags = []
        boundaries = []  # (num_outputs for each question)
        for question, output in zip(questions, outputs):
            processed_data = [prepare_input(question, o, tokenizer=tokenizer, step_token="\n\n") for o in output]
            input_ids, steps, reward_flags = zip(*processed_data)
            flat_input_ids.extend(input_ids)
            flat_reward_flags.extend(reward_flags)
            boundaries.append(len(output))
        prompts = [TokensPrompt(prompt_token_ids=input_id) for input_id in flat_input_ids]
        rewards = self.encode(prompts)
        # Now reconstruct the nested structure
        idx = 0
        for num_outputs in boundaries:
            step_rewards = []
            for _ in range(num_outputs):
                reward = rewards[idx]
                reward_flag = flat_reward_flags[idx]
                reward_embedding = reward.outputs.data.tolist()
                step_reward = []
                for i, flag in enumerate(reward_flag):
                    if flag == 1:
                        step_reward.append(sigmoid(reward_embedding[i][0]))
                step_rewards.append(step_reward)
                idx += 1
            all_scores.append(step_rewards)
        return all_scores