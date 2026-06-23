
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from contextlib import nullcontext

import draccus
import torch
from torch.utils.data import DataLoader

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.act.modeling_act import ACTPolicy


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=device.type == "cuda") if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def load_eval_cfg(checkpoint: Path, eval_root: str, batch_size: int, num_workers: int) -> TrainPipelineConfig:
    pretrained = checkpoint / "pretrained_model" if (checkpoint / "pretrained_model").exists() else checkpoint
    cfg_path = pretrained / "train_config.json"
    raw = json.loads(cfg_path.read_text())
    cfg = draccus.decode(TrainPipelineConfig, raw)
    cfg.dataset.repo_id = eval_root
    cfg.dataset.root = None
    cfg.dataset.episodes = None
    cfg.dataset.video_backend = "pyav"
    cfg.batch_size = batch_size
    cfg.num_workers = num_workers
    cfg.policy.pretrained_path = str(pretrained)
    cfg.policy.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint dir or pretrained_model dir")
    parser.add_argument("--eval-dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=200, help="Use 0 for full dataset")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    pretrained = checkpoint / "pretrained_model" if (checkpoint / "pretrained_model").exists() else checkpoint
    eval_root = str(Path(args.eval_dataset_root).resolve())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_eval_cfg(checkpoint, eval_root, args.batch_size, args.num_workers)
    device = torch.device(cfg.policy.device)
    dataset = make_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    policy = ACTPolicy.from_pretrained(str(pretrained))
    policy.to(device)
    # Keep train mode so ACT VAE returns KL terms needed by its supervised loss.
    # torch.no_grad() below prevents parameter updates.
    policy.train()

    rows = []
    n_batches = 0
    n_frames = 0
    loss_sum = 0.0
    start = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            if args.max_batches and n_batches >= args.max_batches:
                break
            batch = move_to_device(batch, device)
            with torch.autocast(device_type=device.type) if getattr(policy.config, "use_amp", False) else nullcontext():
                loss, _ = policy.forward(batch)
            bs = int(next(v for v in batch.values() if isinstance(v, torch.Tensor)).shape[0])
            n_batches += 1
            n_frames += bs
            loss_value = float(loss.item())
            loss_sum += loss_value * bs
            rows.append({"batch": n_batches, "frames": n_frames, "loss": loss_value})
            if n_batches % 20 == 0:
                print(f"batch={n_batches} frames={n_frames} loss={loss_value:.6f}", flush=True)

    elapsed = time.perf_counter() - start
    metrics = {
        "checkpoint": str(checkpoint),
        "pretrained_model": str(pretrained),
        "eval_dataset_root": eval_root,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_batches": args.max_batches,
        "batches": n_batches,
        "frames": n_frames,
        "mean_loss": loss_sum / max(n_frames, 1),
        "elapsed_s": elapsed,
        "frames_per_s": n_frames / elapsed if elapsed > 0 else None,
    }

    with (output_dir / "offline_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (output_dir / "offline_loss_by_batch.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["batch", "frames", "loss"])
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
