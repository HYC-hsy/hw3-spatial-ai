#!/usr/bin/env bash
set -e
export HW3_ROOT=/mnt/data/kw/hy/projects/course/DL/hw3
source "$HW3_ROOT/.env.hw3_setup"
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh
conda activate "$HW3_ROOT/.conda_envs/threestudio"
export TCNN_CUDA_ARCHITECTURES=80
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$HW3_ROOT/external/bin:$PATH"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "threestudio env ready at $HW3_ROOT"
echo "launch.py: $HW3_ROOT/external/threestudio/launch.py"

export LD_LIBRARY_PATH="$HW3_ROOT/.conda_envs/threestudio/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
