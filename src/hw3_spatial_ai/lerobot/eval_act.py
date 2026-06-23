from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from hw3_spatial_ai.config import ensure_dir, require_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-dataset-root", default="/mnt/data/kw/hy/projects/course/DL/hw3/processed/calvin_lerobot/env_d_eval")
    parser.add_argument("--output-dir", default="/mnt/data/kw/hy/projects/course/DL/hw3/outputs/eval_env_d")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoint = require_path(args.checkpoint, "checkpoint")
    eval_dataset = require_path(args.eval_dataset_root, "eval_dataset_root")
    output_dir = ensure_dir(args.output_dir)

    cmd = [
        "lerobot-eval",
        f"--policy.path={checkpoint}",
        f"--dataset.repo_id={eval_dataset}",
        f"--output_dir={output_dir}",
        f"--eval.n_episodes={args.episodes}",
        f"--device={args.device}",
    ]
    print("Command:")
    print(" ".join(shlex.quote(x) for x in cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()

