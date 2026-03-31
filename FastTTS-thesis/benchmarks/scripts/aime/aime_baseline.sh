export CUDA_VISIBLE_DEVICES=0
python run_benchmarks.py configs/1.5B-1.5B/aime/baseline/aime2024_8.yaml
python run_benchmarks.py configs/1.5B-1.5B/aime/baseline/aime2024_16.yaml
python run_benchmarks.py configs/1.5B-1.5B/aime/baseline/aime2024_32.yaml
python run_benchmarks.py configs/1.5B-1.5B/aime/baseline/aime2024_64.yaml
python run_benchmarks.py configs/1.5B-1.5B/aime/baseline/aime2024_128.yaml
python run_benchmarks.py configs/1.5B-1.5B/aime/baseline/aime2024_256.yaml
python run_benchmarks.py configs/1.5B-1.5B/aime/baseline/aime2024_512.yaml

python run_benchmarks.py configs/1.5B-1.5B/amc/baseline/amc2023_8.yaml
python run_benchmarks.py configs/1.5B-1.5B/amc/baseline/amc2023_16.yaml
python run_benchmarks.py configs/1.5B-1.5B/amc/baseline/amc2023_32.yaml
python run_benchmarks.py configs/1.5B-1.5B/amc/baseline/amc2023_64.yaml
python run_benchmarks.py configs/1.5B-1.5B/amc/baseline/amc2023_128.yaml
python run_benchmarks.py configs/1.5B-1.5B/amc/baseline/amc2023_256.yaml
python run_benchmarks.py configs/1.5B-1.5B/amc/baseline/amc2023_512.yaml