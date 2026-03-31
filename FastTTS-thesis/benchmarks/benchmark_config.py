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
    fasttts_config = create_fasttts_config(
        generator_vllm_config={
            "model": generator_config.get("model", "Qwen/Qwen2.5-Math-1.5B-Instruct"),
            "gpu_memory_utilization": generator_gpu_memory_utilization,
            "tensor_parallel_size": generator_config.get("tensor_parallel_size", 1),
            "enable_prefix_caching": generator_config.get("enable_prefix_caching", True),
            "seed": generator_config.get("seed", 42),
        },
        verifier_vllm_config={
            "model": verifier_config.get("model", "peiyi9979/math-shepherd-mistral-7b-prm"),
            "gpu_memory_utilization": verifier_gpu_memory_utilization,
            "tensor_parallel_size": verifier_config.get("tensor_parallel_size", 1),
            "enable_prefix_caching": verifier_config.get("enable_prefix_caching", True),
            "seed": verifier_config.get("seed", 42),
        },
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