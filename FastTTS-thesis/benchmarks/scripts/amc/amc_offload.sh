export CUDA_VISIBLE_DEVICES=2
python run_benchmarks.py configs/1.5B/amc/offload/amc2023_8.yaml
python run_benchmarks.py configs/1.5B/amc/offload/amc2023_16.yaml
python run_benchmarks.py configs/1.5B/amc/offload/amc2023_32.yaml
python run_benchmarks.py configs/1.5B/amc/offload/amc2023_64.yaml
python run_benchmarks.py configs/1.5B/amc/offload/amc2023_128.yaml
python run_benchmarks.py configs/1.5B/amc/offload/amc2023_256.yaml
python run_benchmarks.py configs/1.5B/amc/offload/amc2023_512.yaml