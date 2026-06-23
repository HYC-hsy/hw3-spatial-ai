#!/usr/bin/env bash
set -euo pipefail

ROOT="${HW3_ROOT:-/mnt/data/kw/hy/projects/course/DL/hw3}"
CODE="${HW3_CODE_ROOT:-$(pwd)}"
export PYTHONPATH="${CODE}/src:${PYTHONPATH:-}"

echo "Code root: ${CODE}"
echo "Server root: ${ROOT}"

python -m py_compile \
  src/hw3_spatial_ai/config.py \
  src/hw3_spatial_ai/calvin/inspect_dataset.py \
  src/hw3_spatial_ai/calvin/prepare_lerobot_splits.py \
  src/hw3_spatial_ai/lerobot/launch_act.py \
  src/hw3_spatial_ai/lerobot/eval_act.py \
  src/hw3_spatial_ai/lerobot/summarize_runs.py \
  src/hw3_spatial_ai/scene/find_mipnerf_scene.py \
  src/hw3_spatial_ai/scene/write_blender_fusion.py

python -m hw3_spatial_ai.scene.write_blender_fusion \
  --config configs/fusion_scene.yaml \
  --out /tmp/hw3_fuse_and_render_check.py

echo "Project check passed."
