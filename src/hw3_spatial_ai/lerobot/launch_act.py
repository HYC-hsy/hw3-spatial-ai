from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from hw3_spatial_ai.config import ensure_dir, load_yaml, require_path


def build_train_command(cfg: dict, dry_run: bool = False) -> list[str]:
    dataset_root = require_path(cfg["dataset_root"], "dataset_root")
    output_dir = Path(cfg["output_dir"])

    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={dataset_root}",
        f"--dataset.video_backend={cfg.get('video_backend', 'pyav')}",
        f"--policy.type={cfg.get('policy', 'act')}",
        f"--output_dir={output_dir}",
        f"--job_name={cfg.get('experiment_name', output_dir.name)}",
        f"--policy.device={cfg.get('device', 'cuda')}",
        f"--batch_size={cfg.get('batch_size', 32)}",
        f"--num_workers={cfg.get('num_workers', 8)}",
        f"--steps={cfg.get('steps', 50000)}",
        f"--optimizer.lr={cfg.get('learning_rate', 1e-5)}",
        f"--optimizer.weight_decay={cfg.get('weight_decay', 1e-4)}",
        f"--policy.chunk_size={cfg.get('chunk_size', 100)}",
        f"--wandb.project={cfg.get('wandb_project', 'dl-hw3-act-calvin')}",
        f"--wandb.mode={cfg.get('wandb_mode', 'online')}",
        f"--policy.push_to_hub={str(cfg.get('push_to_hub', False)).lower()}",
    ]
    for x in cfg.get("extra_args", []):
        arg = str(x)
        if arg and not arg.startswith("-"):
            arg = f"--{arg}"
        cmd.append(arg)
    if dry_run:
        cmd.append("--dry_run=true")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cmd = build_train_command(cfg, dry_run=args.dry_run)
    print("Command:")
    print(" ".join(shlex.quote(x) for x in cmd))
    if args.dry_run:
        return

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()

