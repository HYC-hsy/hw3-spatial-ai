#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/run_zero123_image3d.sh assets/object_c/input.png [steps]"
  exit 2
fi

IMAGE="$1"
STEPS="${2:-600}"
ROOT="${HW3_ROOT:-/mnt/data/kw/hy/projects/course/DL/hw3}"
THREESTUDIO="${ROOT}/external/threestudio"
GPU="${GPU:-0}"

source "${ROOT}/scripts/env_threestudio.sh" >/dev/null

ABS_IMAGE="$(python - <<PY2
from pathlib import Path
print(Path('${IMAGE}').resolve())
PY2
)"
OUT="${ROOT}/outputs/image_asset_c_real"
mkdir -p "${OUT}" "${THREESTUDIO}/load/zero123"

if [ ! -f "${THREESTUDIO}/load/zero123/zero123-xl.ckpt" ]; then
  echo "Missing ${THREESTUDIO}/load/zero123/zero123-xl.ckpt"
  echo "Download it first with: cd ${THREESTUDIO}/load/zero123 && wget https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt"
  exit 4
fi

cd "${THREESTUDIO}"
python launch.py   --config configs/zero123.yaml   --train   --gpu "${GPU}"   exp_root_dir="${OUT}"   name=object_c_zero123   tag=chair_image   data.image_path="${ABS_IMAGE}"   trainer.max_steps="${STEPS}"

CKPT="$(find "${OUT}" -path '*/ckpts/last.ckpt' -type f | sort | tail -n 1)"
if [ -z "${CKPT}" ]; then
  echo "No checkpoint found under ${OUT}"
  exit 3
fi

python launch.py   --config configs/zero123.yaml   --export   --gpu "${GPU}"   resume="${CKPT}"   exp_root_dir="${OUT}"   name=object_c_zero123   tag=chair_image   data.image_path="${ABS_IMAGE}"   system.exporter.fmt=obj-mtl   system.exporter.save_uv=false   system.exporter.save_texture=false   system.exporter.context_type=cuda
