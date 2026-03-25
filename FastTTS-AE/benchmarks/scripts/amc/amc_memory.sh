export CUDA_VISIBLE_DEVICES=3
python run_benchmarks.py configs/7B-1.5B/amc/memory/amc2023_8.yaml
python run_benchmarks.py configs/7B-1.5B/amc/memory/amc2023_16.yaml
python run_benchmarks.py configs/7B-1.5B/amc/memory/amc2023_32.yaml
python run_benchmarks.py configs/7B-1.5B/amc/memory/amc2023_64.yaml
python run_benchmarks.py configs/7B-1.5B/amc/memory/amc2023_128.yaml
python run_benchmarks.py configs/7B-1.5B/amc/memory/amc2023_256.yaml
python run_benchmarks.py configs/7B-1.5B/amc/memory/amc2023_512.yaml

python run_benchmarks.py configs/7B-1.5B/amc/spec_prefix/amc2023_8.yaml
python run_benchmarks.py configs/7B-1.5B/amc/spec_prefix/amc2023_16.yaml
python run_benchmarks.py configs/7B-1.5B/amc/spec_prefix/amc2023_32.yaml
python run_benchmarks.py configs/7B-1.5B/amc/spec_prefix/amc2023_64.yaml
python run_benchmarks.py configs/7B-1.5B/amc/spec_prefix/amc2023_128.yaml
python run_benchmarks.py configs/7B-1.5B/amc/spec_prefix/amc2023_256.yaml
python run_benchmarks.py configs/7B-1.5B/amc/spec_prefix/amc2023_512.yaml
