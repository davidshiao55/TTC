#!/bin/bash
# Incremental rebuild of vllm CUDA/C++ kernels after changes to csrc/.
# Uses ccache — only changed files are recompiled. Much faster than pip install.
# Run with thesis env active: conda activate thesis && ./rebuild_vllm.sh
set -e

cd /TTC/vllm
cmake --build --preset release --target install
echo "vllm rebuild complete."
echo "ccache stats:"
ccache --show-stats
