docker run -it -d --rm \
  --gpus all \
  --ipc=host \
  -v $(pwd):/TTC \
  -w /TTC \
  davidshiao55_ttc_env bash