#!/usr/bin/env bash
set -euo pipefail

ROOT="${HW3_ROOT:-/mnt/data/kw/hy/projects/course/DL/hw3}"
CODE="${HW3_CODE_ROOT:-${ROOT}}"
THREESTUDIO="${ROOT}/external/threestudio"
CONFIG="${CODE}/configs/text_to_3d.yaml"
STEPS="${1:-10000}"
GPU="${GPU:-0}"
MODEL="${SD_MODEL:-sd-legacy/stable-diffusion-v1-5}"

source "${ROOT}/scripts/env_threestudio.sh" >/dev/null

PROMPT="$(python - <<'PY2'
import yaml
with open('configs/text_to_3d.yaml', 'r', encoding='utf-8') as f:
    print(yaml.safe_load(f)['prompt'])
PY2
)"

OUT="${ROOT}/outputs/text_asset_b_real"
mkdir -p "${OUT}"
cd "${THREESTUDIO}"

python launch.py   --config configs/dreamfusion-sd.yaml   --train   --gpu "${GPU}"   exp_root_dir="${OUT}"   name=object_b_sds   tag=robot_prompt   trainer.max_steps="${STEPS}"   system.prompt_processor.prompt="${PROMPT}"   system.prompt_processor.pretrained_model_name_or_path="${MODEL}"   system.guidance.pretrained_model_name_or_path="${MODEL}"

CKPT="$(find "${OUT}" -path '*/ckpts/last.ckpt' -type f | sort | tail -n 1)"
if [ -z "${CKPT}" ]; then
  echo "No checkpoint found under ${OUT}"
  exit 3
fi

python launch.py   --config configs/dreamfusion-sd.yaml   --export   --gpu "${GPU}"   resume="${CKPT}"   exp_root_dir="${OUT}"   name=object_b_sds   tag=robot_prompt   system.prompt_processor.prompt="${PROMPT}"   system.prompt_processor.pretrained_model_name_or_path="${MODEL}"   system.guidance.pretrained_model_name_or_path="${MODEL}"   system.exporter.fmt=obj-mtl   system.exporter.save_uv=false   system.exporter.save_texture=false   system.exporter.context_type=cuda
