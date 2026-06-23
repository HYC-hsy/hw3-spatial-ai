#!/usr/bin/env bash
set -euo pipefail
export HW3_ROOT=/mnt/data/kw/hy/projects/course/DL/hw3
source "$HW3_ROOT/.env.hw3_setup"
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh
conda activate "$HW3_ROOT/.conda_envs/colmap_blender"
echo '[COLMAP]'
colmap -h | head -3
echo '[Blender]'
LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" "$HW3_ROOT/external/bin/blender" --background --version | head -3
conda activate "$HW3_ROOT/.conda_envs/threestudio"
export TCNN_CUDA_ARCHITECTURES=80
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$HW3_ROOT/external/bin:$PATH"
echo '[Threestudio]'
python - <<'PY'
import torch, threestudio, tinycudann, nvdiffrast.torch, envlight
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('threestudio', threestudio.__file__)
print('tinycudann/nvdiffrast/envlight ok')
PY
