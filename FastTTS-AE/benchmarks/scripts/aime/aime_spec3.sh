export CUDA_VISIBLE_DEVICES=3
# python run_benchmarks.py configs/1.5B-7B/aime/spec_prefix_no_future/aime2024_8.yaml
# python run_benchmarks.py configs/1.5B-7B/aime/spec_prefix_no_future/aime2024_16.yaml
python run_benchmarks.py configs/7B-1.5B/aime/spec_prefix_no_future/aime2024_32.yaml
# python run_benchmarks.py configs/1.5B-7B/aime/spec_prefix_no_future/aime2024_64.yaml
# python run_benchmarks.py configs/1.5B-7B/aime/spec_prefix_no_future/aime2024_128.yaml
python run_benchmarks.py configs/7B-1.5B/aime/spec_prefix_no_future/aime2024_256.yaml
# python run_benchmarks.py configs/1.5B-7B/aime/spec_prefix_no_future/aime2024_512.yaml
