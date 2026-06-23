#!/usr/bin/env bash
set -euo pipefail

cd "${HW3_CODE_ROOT:-/mnt/data/kw/hy/projects/course/DL/hw3}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
python -m hw3_spatial_ai.lerobot.launch_act --config configs/act_env_abc_real.yaml
