#!/usr/bin/env bash
set -euo pipefail

ROOT="${HW3_ROOT:-/mnt/data/kw/hy/projects/course/DL/hw3}"
CODE="${HW3_CODE_ROOT:-${ROOT}}"
SCRIPT="${ROOT}/outputs/fusion/fuse_and_render.py"

source "${ROOT}/scripts/env_colmap_blender.sh" >/dev/null
cd "${CODE}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
python -m hw3_spatial_ai.scene.write_blender_fusion   --config configs/fusion_scene.yaml   --out "${SCRIPT}"

"${ROOT}/external/bin/blender" --background --python "${SCRIPT}"
