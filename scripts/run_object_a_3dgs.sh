#!/usr/bin/env bash
set -euo pipefail

NAME="${1:-object_a}"
ROOT="${HW3_ROOT:-/mnt/data/kw/hy/projects/course/DL/hw3}"
GS="${ROOT}/external/gaussian-splatting"
SRC="${HW3_CODE_ROOT:-${ROOT}}/assets/${NAME}"
COLMAP_OUT="${ROOT}/processed/${NAME}_colmap"
OUT="${ROOT}/outputs/${NAME}"

if [ ! -d "${GS}" ]; then
  echo "Missing gaussian-splatting repo: ${GS}"
  exit 2
fi

mkdir -p "${COLMAP_OUT}" "${OUT}"

python "${GS}/convert.py" \
  -s "${SRC}" \
  --colmap_executable colmap \
  --resize

python "${GS}/train.py" \
  -s "${SRC}" \
  -m "${OUT}" \
  --iterations 30000

python "${GS}/render.py" -m "${OUT}"

