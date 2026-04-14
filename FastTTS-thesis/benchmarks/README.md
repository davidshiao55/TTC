# FastTTS Benchmark System (YAML Config Only)

This directory contains a benchmark system for FastTTS that is **fully configurable via user-provided YAML files**. There are no hardcoded or predefined configs—**everything is specified in your YAML**.

## Quick Start

### 1. Prepare a YAML Config

See the `configs/` directory for examples. Here is a minimal example:

```yaml
name: math500_beamsearch
output_dir: benchmark_results

dataset:
  name: HuggingFaceH4/MATH-500
  split: test
  limit: 10

enable_spec_diff: false
offload_enabled: false

# Generator model configuration
generator_model:
  model: "Qwen/Qwen2.5-Math-7B-Instruct"
  max_model_len: 2048
  gpu_memory_utilization: 0.2
  tensor_parallel_size: 1
  enable_prefix_caching: true
  seed: 42

# Verifier model configuration
verifier_model:
  model: "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
  max_model_len: 2048
  gpu_memory_utilization: 0.7
  tensor_parallel_size: 1
  enable_prefix_caching: true
  seed: 42

search_config:
  approach: beam_search
  beam_width: 6
  n: 12
  num_iterations: 30
  max_tokens: 1024
  temperature: 0.8
  stop: "\n\n"
```

### 2. Run the Benchmark

```bash
python run_benchmarks.py configs/math500_beamsearch.yaml
```

### 3. Multi-Dataset Example

You can specify a list of configs in one YAML file:

```yaml
- name: math500_beamsearch
  output_dir: benchmark_results
  dataset:
    name: HuggingFaceH4/MATH-500
    split: test
    
  enable_spec_diff: false
  offload_enabled: false
  generator_model:
    model: "Qwen/Qwen2.5-Math-7B-Instruct"
    max_model_len: 2048
    gpu_memory_utilization: 0.2
    tensor_parallel_size: 1
    enable_prefix_caching: true
    seed: 42
  verifier_model:
    model: "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
    max_model_len: 2048
    gpu_memory_utilization: 0.7
    tensor_parallel_size: 1
    enable_prefix_caching: true
    seed: 42
  search_config:
    approach: beam_search
    beam_width: 4
    n: 8
    num_iterations: 10
    max_tokens: 1024
    temperature: 0.8
    stop: "\n\n"

- name: aime2024_specdiff
  output_dir: benchmark_results
  dataset:
    name: HuggingFaceH4/AIME-2024
    split: test
    limit: 3
  enable_spec_diff: true
  offload_enabled: true
  generator_model:
    model: "Qwen/Qwen2.5-Math-7B-Instruct"
    max_model_len: 2048
    gpu_memory_utilization: 0.2
    tensor_parallel_size: 1
    enable_prefix_caching: true
    seed: 42
  verifier_model:
    model: "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
    max_model_len: 2048
    gpu_memory_utilization: 0.4
    tensor_parallel_size: 1
    enable_prefix_caching: true
    seed: 42
  search_config:
    approach: beam_search
    beam_width: 4
    n: 8
    num_iterations: 15
  
    temperature: 0.8
    stop: "\n\n"
```

Run with:
```bash
python run_benchmarks.py configs/multi_dataset.yaml
```

## YAML Config Fields

- `name`: Name for this benchmark run (used in output filename)
- `output_dir`: Directory to save results
- `dataset`:
    - `name`: HuggingFace dataset name
    - `split`: Dataset split (e.g., `test`)
    - `limit`: Max number of problems
- `enable_spec_diff`: Enable speculative diff optimization (bool)
- `offload_enabled`: Enable offloading for both generator and verifier models (bool)
- `generator_model`: Generator model configuration
    - `model`: HuggingFace model name
    - `max_model_len`: Maximum sequence length
    - `gpu_memory_utilization`: GPU memory utilization (0.0-1.0)
    - `tensor_parallel_size`: Tensor parallelism size
    - `enable_prefix_caching`: Enable prefix caching
    - `seed`: Random seed
- `verifier_model`: Verifier model configuration (same fields as generator_model)
- `search_config`:
    - `approach`: Search approach (e.g., `beam_search`)
    - `beam_width`, `n`, `num_iterations`, `max_tokens`, `temperature`, `stop`: Usual search parameters

## Example Configs

See `benchmarks/configs/` for:
- `math500_beamsearch.yaml` - Basic MATH500 with Qwen models
- `aime2023_offload.yaml` - AIME2023 with model offloading and lower GPU memory
- `gsm8k_llama.yaml` - GSM8K with Llama models and different reward model
- `high_memory_utilization.yaml` - High GPU memory utilization example
- `multi_dataset.yaml` - Multiple datasets in one config

## GPU Memory Utilization Tips

- **Low memory (0.1-0.3)**: Use with model offloading, smaller models
- **Medium memory (0.3-0.6)**: Standard settings for most use cases
- **High memory (0.7-0.9)**: For larger models, when you have sufficient GPU memory
- **Memory allocation**: Generator typically uses less memory than verifier

## Files

- `run_benchmarks.py` - Main benchmark runner (YAML config only)
- `benchmark_config.py` - YAML config parser
- `dataset_utils.py` - Dataset loading utilities
- `configs/` - Example YAML configuration files
- `README.md` - This documentation

## Requirements
- FastTTS package installed
- `pyyaml` (`pip install pyyaml`)
- Access to HuggingFace datasets
- GPU with sufficient memory for models

## Output
- Results are saved in the specified `output_dir` with descriptive filenames.

## No More Hardcoded Configs!
- All configuration is now user-driven via YAML.
- You can version, share, and modify your experiment configs easily.
- Full control over model selection and GPU memory allocation.