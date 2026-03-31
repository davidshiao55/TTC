from vllm import LLM, SamplingParams
from collections.abc import Sequence
from typing import (Any, Callable, Optional, Union, cast)
import torch
from collections import deque

import cloudpickle
from tqdm.auto import tqdm

from vllm.config import (CompilationConfig, is_init_field)
from vllm.config.model import ModelDType, TokenizerMode
from vllm.engine.arg_utils import (EngineArgs, HfOverrides, PoolerConfig,
                                   RunnerOption)
from vllm.inputs import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.outputs import (PoolingRequestOutput, RequestOutput)
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils.counter import Counter

from .generator_engine_v1 import GeneratorLLMEngineV1
from vllm.v1.engine.llm_engine import LLMEngine as VerifierLLMEngine

from .reward_utils import prepare_input, sigmoid

import torch.cuda.nvtx as nvtx

logger = init_logger(__name__)

# Custom synchronous LLM wrapper
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

        # SBE requires priority scheduling, which must be set at engine
        # init time (the scheduler queue type is chosen once in __init__).
        if spec_beam_extension:
            kwargs.setdefault("scheduling_policy", "priority")

        # Only pass seed when explicitly provided — thesis vLLM requires
        # int (rejects None).  Omitting lets EngineArgs default to 0.
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
            compilation_config=compilation_config_instance,
            **kwargs,
        )

        # Select engine class
        if runner in ('generate', 'auto'):
            engine_cls = GeneratorLLMEngineV1
            logger.info(f"Using V1 engine with speculative beam extension: {spec_beam_extension}")
        else:
            engine_cls = VerifierLLMEngine
        logger.info(f"Prefix-aware scheduling enabled: {prefix_aware_scheduling}")
        self.llm_engine = engine_cls.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS)
        if spec_beam_extension and isinstance(self.llm_engine, GeneratorLLMEngineV1):
            self.llm_engine.enable_spec_beam_extension()

        self.engine_class = type(self.llm_engine)
        self.request_counter = Counter()
        self.default_sampling_params: Union[dict[str, Any], None] = None
        self.model_path = model  # Store the model path for score_outputs dispatch

        # Set attributes required by parent LLM methods (generate, encode, etc.)
        self.model_config = self.llm_engine.model_config
        self.renderer = getattr(self.llm_engine, 'renderer', None)
        self.runner_type = self.model_config.runner_type
        supported_tasks = self.llm_engine.get_supported_tasks()
        self.supported_tasks = supported_tasks
        self.pooling_task = self.model_config.get_pooling_task(supported_tasks)
        self.pooling_io_processors = {}
        self.io_processor = getattr(self.llm_engine, 'io_processor', None)
        self.input_processor = getattr(self.llm_engine, 'input_processor', None)

    def generate_text(self, prompts, **kwargs):
        assert isinstance(self.llm_engine, GeneratorLLMEngineV1), "GeneratorLLMEngineV1 is required for generate_text"
        request_outputs = self.generate(prompts, **kwargs)
        return request_outputs

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
        """Dispatch to the correct scoring implementation based on model_path."""
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
                if len(reward_embedding) != len(reward_flag):
                    logger.error(
                        f"Shape mismatch: reward_embedding has {len(reward_embedding)} rows "
                        f"but reward_flag has {len(reward_flag)} entries "
                        f"(input_ids had {len(flat_input_ids[idx])} tokens). "
                        f"Prompt idx={idx}"
                    )
                for i, flag in enumerate(reward_flag):
                    if flag == 1:
                        if i >= len(reward_embedding):
                            logger.error(
                                f"Index {i} out of range for reward_embedding "
                                f"(len={len(reward_embedding)}), "
                                f"input_ids len={len(flat_input_ids[idx])}"
                            )
                            break
                        step_reward.append(sigmoid(reward_embedding[i][0]))
                step_rewards.append(step_reward)
                idx += 1
            all_scores.append(step_rewards)
        return all_scores
