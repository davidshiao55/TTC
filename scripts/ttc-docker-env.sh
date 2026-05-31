#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/ttc-docker-env.sh thesis [command]
  scripts/ttc-docker-env.sh baseline [command]

Environment variables:
  TTC_DOCKER_CONTAINER   Container name. Default: davidshiao55_ttc
  TTC_DOCKER_WORKDIR     Directory inside container. Default: /TTC

Examples:
  scripts/ttc-docker-env.sh thesis 'cd /tmp && python -c "import vllm; print(vllm.__version__)"'
  TTC_DOCKER_WORKDIR=/TTC/vllm scripts/ttc-docker-env.sh thesis 'pytest tests/v1/worker/test_cots_hybrid_kv.py -q'
USAGE
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

env_name="$1"
shift

case "$env_name" in
  baseline|thesis) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown conda env: $env_name" >&2
    usage >&2
    exit 2
    ;;
esac

container="${TTC_DOCKER_CONTAINER:-davidshiao55_ttc}"
workdir="${TTC_DOCKER_WORKDIR:-/TTC}"
host_uid="$(id -u)"
host_gid="$(id -g)"
container_home="/tmp/ttc-codex-home-${host_uid}"
host_git_user_name="${TTC_GIT_USER_NAME:-$(git config --global --get user.name 2>/dev/null || true)}"
host_git_user_email="${TTC_GIT_USER_EMAIL:-$(git config --global --get user.email 2>/dev/null || true)}"

docker_args=(
  exec
  --user "${host_uid}:${host_gid}"
  -e USER="${USER:-codex}"
  -e LOGNAME="${LOGNAME:-${USER:-codex}}"
  -e HOME="${container_home}"
  -e XDG_CACHE_HOME="${container_home}/.cache"
  -e TORCHINDUCTOR_CACHE_DIR="${container_home}/.cache/torchinductor"
  -e HF_HOME="${HF_HOME:-/models/huggingface}"
)

if [[ -n "$host_git_user_name" ]]; then
  docker_args+=(-e TTC_HOST_GIT_USER_NAME="$host_git_user_name")
fi

if [[ -n "$host_git_user_email" ]]; then
  docker_args+=(-e TTC_HOST_GIT_USER_EMAIL="$host_git_user_email")
fi

if [[ -t 0 && -t 1 ]]; then
  docker_args+=(-it)
fi

if [[ $# -eq 0 ]]; then
  command='exec bash'
else
  command="$*"
fi

docker "${docker_args[@]}" "$container" bash -lc "
set -euo pipefail
mkdir -p \"\$HOME\" \"\$XDG_CACHE_HOME\" \"\$TORCHINDUCTOR_CACHE_DIR\"
if [[ -n \"\${TTC_HOST_GIT_USER_NAME:-}\" ]] && ! git config --global --get user.name >/dev/null 2>&1; then
  git config --global user.name \"\$TTC_HOST_GIT_USER_NAME\"
fi
if [[ -n \"\${TTC_HOST_GIT_USER_EMAIL:-}\" ]] && ! git config --global --get user.email >/dev/null 2>&1; then
  git config --global user.email \"\$TTC_HOST_GIT_USER_EMAIL\"
fi
source /opt/conda/etc/profile.d/conda.sh
conda activate '$env_name'
cd '$workdir'
$command
"
