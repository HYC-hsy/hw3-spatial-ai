#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/eval_act_env_d.sh /path/to/checkpoint [output_dir]"
  exit 2
fi

CHECKPOINT="$1"
OUT="${2:-/mnt/data/kw/hy/projects/course/DL/hw3/outputs/eval_env_d}"

cd "${HW3_CODE_ROOT:-/mnt/data/kw/hy/projects/course/DL/hw3}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
python -m hw3_spatial_ai.lerobot.eval_act \
  --checkpoint "${CHECKPOINT}" \
  --eval-dataset-root /mnt/data/kw/hy/projects/course/DL/hw3/processed/calvin_lerobot/env_d_eval \
  --output-dir "${OUT}"
