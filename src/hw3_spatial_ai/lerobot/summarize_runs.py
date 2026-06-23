from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def find_metrics(run_dir: Path) -> dict:
    metrics = {"run": run_dir.name}
    for path in run_dir.rglob("*.json"):
        if path.name in {"metrics.json", "eval_info.json", "results.json"}:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            for key, value in data.items():
                if isinstance(value, (int, float, str)):
                    metrics[key] = value
    for path in run_dir.rglob("*.csv"):
        if "progress" in path.name or "metrics" in path.name:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if not df.empty:
                for col in df.columns:
                    if "loss" in col.lower() or "success" in col.lower() or "l1" in col.lower():
                        metrics[f"last_{col}"] = df[col].dropna().iloc[-1] if df[col].dropna().size else None
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    rows = [find_metrics(p) for p in sorted(runs_root.iterdir()) if p.is_dir()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

