export CUDA_VISIBLE_DEVICES=0
python run_benchmarks.py configs/humaneval/baseline/aime2024_8.yaml
python run_benchmarks.py configs/humaneval/baseline/aime2024_16.yaml
python run_benchmarks.py configs/humaneval/baseline/aime2024_32.yaml
python run_benchmarks.py configs/humaneval/baseline/aime2024_64.yaml
python run_benchmarks.py configs/humaneval/baseline/aime2024_128.yaml
python run_benchmarks.py configs/humaneval/baseline/aime2024_256.yaml
python run_benchmarks.py configs/humaneval/baseline/aime2024_512.yaml