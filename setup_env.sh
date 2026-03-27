#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
set -e

# --- Baseline: original FastTTS-AE + vllm 0.9.2 from PyPI ---
echo "=== Creating baseline env ==="
conda create -y -n baseline python=3.11
conda activate baseline
pip install -e /TTC/FastTTS-AE
pip install -e /TTC/FastTTS-AE/modified-skywork-o1-prm-inference
conda deactivate

# --- Thesis: modified FastTTS-thesis + vllm fork ---
# Step 1: precompiled install to resolve all Python dependencies quickly.
# Step 2: cmake full build to compile CUDA kernels (required for CUDA Graph /
#         cudaLaunchHostFunc). ccache makes subsequent incremental rebuilds fast.
# Step 3: install FastTTS-thesis on top.
# For subsequent vllm C/CUDA changes use: ./rebuild_vllm.sh
echo "=== Creating thesis env ==="
conda create -y -n thesis python=3.11
conda activate thesis

# Install build deps and ccache
conda install -y ccache
pip install -r /TTC/vllm/requirements/build.txt

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
echo "  ./rebuild_vllm.sh         # incremental CUDA rebuild after kernel changes"
