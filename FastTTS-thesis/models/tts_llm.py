from vllm import LLM
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
        self.supported_tasks = self.llm_engine.get_supported_tasks()
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

    _score_call_counter = 0  # TODO(debug): remove after investigation

    def _score_outputs_skywork(self, questions, outputs, _, prev_scores=None, skipped_beam_context=None, **kwargs):
        # TODO(debug): remove after investigation
        TTSLLM._score_call_counter += 1
        call_id = TTSLLM._score_call_counter

        all_scores = []
        tokenizer = self.get_tokenizer()
        flat_input_ids = []
        flat_reward_flags = []
        flat_prev_scores = []
        boundaries = []  # (num_outputs for each question)
        prev_idx = 0
        for question, output in zip(questions, outputs):
            processed_data = [prepare_input(question, o, tokenizer=tokenizer, step_token="\n\n") for o in output]
            input_ids, steps, reward_flags = zip(*processed_data)
            flat_input_ids.extend(input_ids)
            flat_reward_flags.extend(reward_flags)
            for _ in output:
                if prev_scores is not None and prev_idx < len(prev_scores):
                    flat_prev_scores.append(prev_scores[prev_idx])
                else:
                    flat_prev_scores.append([])
                prev_idx += 1
            boundaries.append(len(output))
        prompts = [TokensPrompt(prompt_token_ids=input_id) for input_id in flat_input_ids]
        for i, ids in enumerate(flat_input_ids):
            if len(ids) > 4096:
                logger.error(
                    f"prepare_input returned {len(ids)} tokens > 4096 for prompt {i}. "
                    f"Question len={len(tokenizer.encode(tokenizer.bos_token + questions[0] + chr(10)))}, "
                    f"response preview=...{str(ids[-10:])}"
                )
        rewards = self.encode(
            prompts,
            pooling_params=PoolingParams(skip_reading_prefix_cache=False),
        )
        # Reconstruct the nested structure.
        # With prefix caching, reward_embedding may be shorter than reward_flag
        # because cached prefix tokens skip the forward pass. Use None
        # placeholders for missing scores, then fill from prev_scores and
        # within-batch propagation.
        flat_step_rewards = []
        idx = 0
        for num_outputs in boundaries:
            for _ in range(num_outputs):
                reward = rewards[idx]
                reward_flag = flat_reward_flags[idx]
                reward_embedding = reward.outputs.data.tolist()
                offset = len(reward_flag) - len(reward_embedding)
                if offset < 0:
                    logger.error(
                        f"reward_embedding ({len(reward_embedding)}) larger than "
                        f"reward_flag ({len(reward_flag)}) — unexpected. "
                        f"Prompt idx={idx}"
                    )
                step_reward = []
                for i, flag in enumerate(reward_flag):
                    if flag == 1:
                        local_idx = i - offset
                        if local_idx < 0:
                            step_reward.append(None)  # cached — fill later
                        elif local_idx >= len(reward_embedding):
                            break
                        else:
                            step_reward.append(sigmoid(reward_embedding[local_idx][0]))
                flat_step_rewards.append(step_reward)
                idx += 1

        # Layer 1: Prefer prev_scores over fresh computation.
        # Once a step is scored, its value is locked — fresh recomputation
        # is ignored to ensure consistent scores across iterations and
        # duplicates (avoids BF16 noise from different computation contexts).
        for idx, step_reward in enumerate(flat_step_rewards):
            prev = flat_prev_scores[idx]
            for j in range(min(len(step_reward), len(prev))):
                if prev[j] is not None:
                    step_reward[j] = prev[j]

        # Layer 2: Within-batch propagation (fill from batch + skipped beams)
        has_nones = any(None in sr for sr in flat_step_rewards)
        if has_nones:
            self._propagate_scores_within_batch(
                flat_step_rewards, flat_input_ids, flat_reward_flags,
                skipped_beam_context=skipped_beam_context,
                tokenizer=tokenizer,
            )

        # Layer 3: Remaining Nones indicate a bug — abort with diagnostics
        for idx, step_reward in enumerate(flat_step_rewards):
            if any(s is None for s in step_reward):
                n_missing = sum(1 for s in step_reward if s is None)
                reward_flag = flat_reward_flags[idx]
                reward_embedding = rewards[idx].outputs.data.tolist()
                offset = len(reward_flag) - len(reward_embedding)
                flag_positions = [i for i, f in enumerate(reward_flag) if f == 1]
                none_positions = [j for j, s in enumerate(step_reward) if s is None]
                prev = flat_prev_scores[idx] if idx < len(flat_prev_scores) else []
                problem_text = tokenizer.decode(flat_input_ids[idx])
                raise RuntimeError(
                    f"PRM score propagation failed at call {call_id}: "
                    f"{n_missing}/{len(step_reward)} step scores still None "
                    f"(prompt idx={idx}).\n"
                    f"  offset={offset}, input_ids len={len(flat_input_ids[idx])}\n"
                    f"  flag_positions={flag_positions}\n"
                    f"  none_positions={none_positions}\n"
                    f"  prev_scores len={len(prev)}\n"
                    f"\nFULL TEXT:\n{problem_text}"
                )

        # TODO(debug): dump post-merge state to debug file
        import os as _os
        _debug_path = _os.path.join(_os.getcwd(), "experiments/results/score_debug_dump.txt")
        with open(_debug_path, "a") as _dbg:
            _dbg.write(f"\n{'='*80}\n")
            _dbg.write(f"Score call {call_id}: {len(flat_step_rewards)} beams (POST-MERGE)\n")
            _dbg.write(f"{'='*80}\n")
            for b_idx, sr in enumerate(flat_step_rewards):
                ids = flat_input_ids[b_idx]
                emb_len = len(rewards[b_idx].outputs.data.tolist())
                offset = len(ids) - emb_len
                flags = [i for i, f in enumerate(flat_reward_flags[b_idx]) if f == 1]
                prev = flat_prev_scores[b_idx]
                _dbg.write(
                    f"\n--- beam {b_idx}: tokens={len(ids)} cached={offset} "
                    f"computed={emb_len} prev={len(prev)} ---\n"
                )
                _step_bounds = [0] + flags + [len(ids)]
                for j in range(len(_step_bounds) - 1):
                    start, end = _step_bounds[j], _step_bounds[j + 1]
                    step_text = tokenizer.decode(ids[start:end])
                    if end <= offset:
                        region = "CACHED"
                    elif start >= offset:
                        region = "COMPUTED"
                    else:
                        region = f"SPLIT@{offset}"
                    if j < len(sr):
                        if sr[j] is None:
                            score = "NONE"
                        else:
                            # Show source: prev, propagated, or computed
                            in_cached = (flags[j] < offset) if j < len(flags) else False
                            if in_cached and j < len(prev) and prev[j] is not None:
                                score = f"{sr[j]:.4f} (from prev)"
                            elif in_cached:
                                score = f"{sr[j]:.4f} (propagated)"
                            else:
                                score = f"{sr[j]:.4f} (computed)"
                    else:
                        score = "n/a"
                    _dbg.write(
                        f"  [step {j}] tokens {start}-{end} ({region}) "
                        f"score={score}\n"
                        f"    {step_text}\n"
                    )
        # END TODO(debug)

        # Rebuild nested structure
        idx = 0
        for num_outputs in boundaries:
            step_rewards = []
            for _ in range(num_outputs):
                step_rewards.append(flat_step_rewards[idx])
                idx += 1
            all_scores.append(step_rewards)
        return all_scores

    @staticmethod
    def _propagate_scores_within_batch(flat_step_rewards, flat_input_ids, flat_reward_flags,
                                       skipped_beam_context=None, tokenizer=None):
        """Copy missing scores from batch neighbors and skipped beams.

        If beam A and beam B share the same token prefix up to a step boundary,
        then the PRM score at that boundary is identical (same tokens → same
        hidden state). This lets us copy scores from whichever beam computed them.

        Skipped beams (SBE skip logic) are not in the current scoring batch
        but their scores are available via skipped_beam_context. Their entries
        are added to the bank so sibling beams can find matching prefixes.

        Only runs when Nones exist (not on the critical path).
        """
        bank = {}

        # Add entries from skipped beams (not in current batch but have scores)
        if skipped_beam_context and tokenizer:
            for question, completion, all_scores in skipped_beam_context:
                input_ids, _, reward_flags = prepare_input(
                    question, completion, tokenizer=tokenizer, step_token="\n\n"
                )
                flag_positions = [i for i, f in enumerate(reward_flags) if f == 1]
                for j, pos in enumerate(flag_positions):
                    if j < len(all_scores) and all_scores[j] is not None:
                        prefix_key = tuple(input_ids[:pos + 1])
                        bank[prefix_key] = all_scores[j]

        # Add entries from current batch
        for idx, step_scores in enumerate(flat_step_rewards):
            flags = flat_reward_flags[idx]
            ids = flat_input_ids[idx]
            flag_positions = [i for i, f in enumerate(flags) if f == 1]
            for j, pos in enumerate(flag_positions):
                if j < len(step_scores) and step_scores[j] is not None:
                    prefix_key = tuple(ids[:pos + 1])
                    bank[prefix_key] = step_scores[j]

        # Fill Nones from bank
        for idx, step_scores in enumerate(flat_step_rewards):
            if None not in step_scores:
                continue
            flags = flat_reward_flags[idx]
            ids = flat_input_ids[idx]
            flag_positions = [i for i, f in enumerate(flags) if f == 1]
            for j, pos in enumerate(flag_positions):
                if j < len(step_scores) and step_scores[j] is None:
                    prefix_key = tuple(ids[:pos + 1])
                    if prefix_key in bank:
                        step_scores[j] = bank[prefix_key]
