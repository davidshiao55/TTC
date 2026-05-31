#!/usr/bin/env bash
set -euo pipefail

container="${TTC_DOCKER_CONTAINER:-davidshiao55_ttc}"
image="${TTC_DOCKER_IMAGE:-davidshiao55_ttc_env}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_dir="${TTC_MODELS_DIR:-/home/davidshiao55/models}"

docker run -it -d \
  --name "$container" \
  --gpus all \
  --ipc=host \
  -v "$repo_root":/TTC \
  -v "$models_dir":/models \
  -e HF_HOME=/models/huggingface \
  -w /TTC \
  "$image" bash
