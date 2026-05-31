#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh

configure_git_safe_directories() {
    # setup-env runs as root inside Docker, while /TTC is host-owned. vLLM's
    # editable build asks Git for version metadata, so Git must trust the mount.
    local dir
    for dir in /TTC /TTC/vllm; do
        git config --global --add safe.directory "$dir" || true
    done
}

conda_env_exists() {
    conda env list | awk '{print $1}' | grep -qx "$1"
}

create_or_reuse_env() {
    local env_name="$1"
    if conda_env_exists "$env_name"; then
        echo "=== Reusing existing $env_name env ==="
    else
        conda create -y -n "$env_name" python=3.11
    fi
}

configure_git_safe_directories

# --- Baseline: original FastTTS-AE + vllm 0.9.2 from PyPI ---
echo "=== Creating baseline env ==="
create_or_reuse_env baseline
conda activate baseline
pip install -e /TTC/FastTTS-AE
pip install -e /TTC/FastTTS-AE/modified-skywork-o1-prm-inference
# latex2sympy must be installed from local source (accuracy evaluation dep)
pip install -e /TTC/FastTTS-AE/accuracy_evaluation/evaluation/latex2sympy
conda deactivate

# --- Thesis: modified FastTTS-thesis + vllm fork ---
# Step 1: precompiled install to resolve all Python dependencies quickly.
# Step 2: cmake full build to compile CUDA kernels (required for CUDA Graph /
#         cudaLaunchHostFunc). ccache makes subsequent incremental rebuilds fast.
# Step 3: install FastTTS-thesis on top.
# For subsequent vllm C/CUDA changes use: /TTC/scripts/rebuild-vllm.sh
echo "=== Creating thesis env ==="
create_or_reuse_env thesis
conda activate thesis

# Install build deps and ccache
conda install -y ccache
pip install -r /TTC/vllm/requirements/build.txt
# Minimal test/lint harness deps for recorded TTC/vLLM validation and hooks.
# Avoid installing vLLM's full requirements/test.txt here; it is much broader
# than the thesis smoke/regression checks need.
pip install pytest pytest-forked tblib pre-commit

# Python-level install (fast — precompiled .so, no cmake)
VLLM_USE_PRECOMPILED=1 pip install -e /TTC/vllm

# Generate CMakeUserPresets.json (auto-detects nvcc, python, cpu cores)
python /TTC/vllm/tools/generate_cmake_presets.py --force-overwrite

# Full CUDA build — replaces precompiled .so with our fork's kernels
# cmake presets must be run from the vllm root (where CMakeUserPresets.json lives)
cd /TTC/vllm
cmake --preset release
cmake --build --preset release --target install
cd -

pip install -e /TTC/FastTTS-thesis
pip install -e /TTC/FastTTS-thesis/modified-skywork-o1-prm-inference
pip install -e /TTC/FastTTS-thesis/accuracy_evaluation/evaluation/latex2sympy
conda deactivate

echo "=== Verifying ==="
conda run -n baseline python -c "import vllm; print('baseline vllm:', vllm.__version__)"
# Run from /tmp: CWD='' in sys.path must not contain a 'vllm' dir or PathFinder
# wins over the editable finder with a namespace package (the repo root /TTC/vllm).
(cd /tmp && conda run -n thesis python -c "import vllm; print('thesis   vllm:', vllm.__version__, '|', vllm.__spec__.submodule_search_locations[0])")

echo ""
echo "Usage:"
echo "  conda activate baseline   # original FastTTS + vllm 0.9.2"
echo "  conda activate thesis     # modified FastTTS + vllm fork"
echo "  /TTC/scripts/rebuild-vllm.sh  # incremental CUDA rebuild after kernel changes"
