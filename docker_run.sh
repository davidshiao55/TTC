docker run -it -d \
  --name davidshiao55_ttc \
  --gpus all \
  --ipc=host \
  -v $(pwd):/TTC \
  -v /home/davidshiao55/models:/models \
  -e HF_HOME=/models/huggingface \
  -w /TTC \
  davidshiao55_ttc_env bash
