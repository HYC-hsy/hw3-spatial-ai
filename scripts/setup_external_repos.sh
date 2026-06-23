#!/usr/bin/env bash
set -euo pipefail

ROOT="${HW3_ROOT:-/mnt/data/kw/hy/projects/course/DL/hw3}"
EXTERNAL="${ROOT}/external"
mkdir -p "${EXTERNAL}"

cd "${EXTERNAL}"

clone_if_missing() {
  local name="$1"
  local url="$2"
  if [ ! -d "${name}/.git" ]; then
    git clone "${url}" "${name}"
  else
    echo "Already exists: ${EXTERNAL}/${name}"
  fi
}

clone_if_missing gaussian-splatting https://github.com/graphdeco-inria/gaussian-splatting.git
clone_if_missing threestudio https://github.com/threestudio-project/threestudio.git
clone_if_missing zero123 https://github.com/cvlab-columbia/zero123.git
clone_if_missing lerobot https://github.com/huggingface/lerobot.git

echo "External repositories are under ${EXTERNAL}"

