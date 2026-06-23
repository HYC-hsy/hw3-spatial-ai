#!/usr/bin/env bash
set -e
export HW3_ROOT=/mnt/data/kw/hy/projects/course/DL/hw3
source "$HW3_ROOT/.env.hw3_setup"
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh
conda activate "$HW3_ROOT/.conda_envs/colmap_blender"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PATH="$HW3_ROOT/external/bin:$PATH"
echo "COLMAP/Blender env ready at $HW3_ROOT"
echo "colmap: $(command -v colmap)"
echo "blender: $HW3_ROOT/external/bin/blender"
