from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from fasttts import SearchConfig, FastTTSConfig, create_fasttts_config

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run, loaded from YAML."""
    name: str
    dataset_name: str
    dataset_split: str = "test"
    dataset_subset: Optional[str] = None
    output_dir: str = "benchmark_results"
    fasttts_config: FastTTSConfig = field(default_factory=FastTTSConfig)
    search_config: SearchConfig = field(default_factory=SearchConfig)
    limit: Optional[int] = None
    prompt_field: Optional[str] = None
    answer_field: Optional[str] = None
    # Optimization techniques
    enable_spec_diff: bool = False
    offload_enabled: bool = False
    enable_prefix_aware_scheduling: bool = False


def build_benchmark_config_from_yaml(yaml_dict: Dict[str, Any]) -> BenchmarkConfig:
    """Build a BenchmarkConfig from a YAML dictionary."""
    # Extract optimization options
    enable_spec_diff = yaml_dict.get("enable_spec_diff", False)
    offload_enabled = yaml_dict.get("offload_enabled", False)
    enable_prefix_aware_scheduling = yaml_dict.get("enable_prefix_aware_scheduling", False)

    # Dataset
    dataset = yaml_dict["dataset"]
    dataset_name = dataset["name"]
    dataset_split = dataset.get("split", "test")
    dataset_subset = dataset.get("subset")
    limit = dataset.get("limit")
    prompt_field = dataset.get("prompt_field")
    answer_field = dataset.get("answer_field")

    # Search config
    search_cfg = yaml_dict["search_config"]
    search_config = SearchConfig(**search_cfg)

    # Model configurations
    generator_config = yaml_dict.get("generator_model", {})
    verifier_config = yaml_dict.get("verifier_model", {})
    
    # GPU memory utilization
    generator_gpu_memory_utilization = generator_config.get("gpu_memory_utilization", 0.45)
    verifier_gpu_memory_utilization = verifier_config.get("gpu_memory_utilization", 0.45)
    
    # Build FastTTS config with model settings
    # Build vllm configs — pass through max_model_len if specified
    gen_vllm_config = {
        "model": generator_config.get("model", "Qwen/Qwen2.5-Math-1.5B-Instruct"),
        "gpu_memory_utilization": generator_gpu_memory_utilization,
        "tensor_parallel_size": generator_config.get("tensor_parallel_size", 1),
        "enable_prefix_caching": generator_config.get("enable_prefix_caching", True),
        "seed": generator_config.get("seed", 42),
    }
    if "max_model_len" in generator_config:
        gen_vllm_config["max_model_len"] = generator_config["max_model_len"]
    if "kv_offloading_size" in generator_config:
        gen_vllm_config["kv_offloading_size"] = generator_config["kv_offloading_size"]
        gen_vllm_config["kv_offloading_backend"] = generator_config.get(
            "kv_offloading_backend", "native"
        )
        # vLLM v1 has an ordering bug in VllmConfig.__post_init__: the HMA
        # auto-disable check runs before `_post_init_kv_transfer_config()`
        # materialises `kv_transfer_config` from `kv_offloading_size`, so
        # HMA stays on and OffloadingConnector's factory rejects it. Set
        # this explicitly to bypass the auto-detect path.
        gen_vllm_config["disable_hybrid_kv_cache_manager"] = True

    ver_vllm_config = {
        "model": verifier_config.get("model", "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"),
        "gpu_memory_utilization": verifier_gpu_memory_utilization,
        "tensor_parallel_size": verifier_config.get("tensor_parallel_size", 1),
        "enable_prefix_caching": verifier_config.get("enable_prefix_caching", True),
        "seed": verifier_config.get("seed", 42),
    }
    if "max_model_len" in verifier_config:
        ver_vllm_config["max_model_len"] = verifier_config["max_model_len"]
    if "kv_offloading_size" in verifier_config:
        ver_vllm_config["kv_offloading_size"] = verifier_config["kv_offloading_size"]
        ver_vllm_config["kv_offloading_backend"] = verifier_config.get(
            "kv_offloading_backend", "native"
        )
        ver_vllm_config["disable_hybrid_kv_cache_manager"] = True

    fasttts_config = create_fasttts_config(
        generator_vllm_config=gen_vllm_config,
        verifier_vllm_config=ver_vllm_config,
        offload_enabled=offload_enabled,
        spec_beam_extension=enable_spec_diff,
        prefix_aware_scheduling=enable_prefix_aware_scheduling,
    )

    return BenchmarkConfig(
        name=yaml_dict.get("name", dataset_name),
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dataset_subset=dataset_subset,
        output_dir=yaml_dict.get("output_dir", "benchmark_results"),
        fasttts_config=fasttts_config,
        search_config=search_config,
        limit=limit,
        prompt_field=prompt_field,
        answer_field=answer_field,
        enable_spec_diff=enable_spec_diff,
        offload_enabled=offload_enabled,
        enable_prefix_aware_scheduling=enable_prefix_aware_scheduling,
    )