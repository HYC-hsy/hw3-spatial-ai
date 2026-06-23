from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_tree(root: Path, max_items: int = 80) -> dict:
    suffix_counts: dict[str, int] = {}
    examples: list[str] = []
    has_lerobot_meta = (root / "meta").exists()
    has_lerobot_data = (root / "data").exists()

    for path in root.rglob("*"):
        if path.is_file():
            suffix = path.suffix.lower() or "<none>"
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
            if len(examples) < max_items:
                examples.append(str(path.relative_to(root)))

    return {
        "root": str(root),
        "exists": root.exists(),
        "looks_like_lerobot": has_lerobot_meta and has_lerobot_data,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--max-items", type=int, default=80)
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        raise FileNotFoundError(f"CALVIN data root not found: {root}")

    report = inspect_tree(root, args.max_items)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if report["looks_like_lerobot"]:
        print("\nDetected LeRobot-style dataset with data/ and meta/.")
    else:
        print("\nDid not detect a ready LeRobotDataset. Run prepare_lerobot_splits.py or adapt the converter.")


if __name__ == "__main__":
    main()

