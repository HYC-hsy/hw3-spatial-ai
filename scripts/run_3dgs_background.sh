#!/usr/bin/env bash
set -euo pipefail

SCENE="${1:-flowers}"
ROOT="${HW3_ROOT:-/mnt/data/kw/hy/projects/course/DL/hw3}"
GS="${ROOT}/external/gaussian-splatting"
DATA="${ROOT}/data/360_extra_scenes/${SCENE}"
OUT="${ROOT}/outputs/3dgs_background/${SCENE}"

source "${ROOT}/scripts/env_threestudio.sh"

if [ ! -d "${GS}" ]; then
  echo "Missing gaussian-splatting repo: ${GS}"
  echo "Run: bash scripts/setup_external_repos.sh"
  exit 2
fi

python "${GS}/train.py" \
  -s "${DATA}" \
  -m "${OUT}" \
  --iterations 30000

python "${GS}/render.py" -m "${OUT}"
python "${GS}/metrics.py" -m "${OUT}"
