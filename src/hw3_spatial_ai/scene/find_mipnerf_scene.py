from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--scene", required=True)
    args = parser.parse_args()

    root = Path(args.data_root)
    scene = root / args.scene
    if not scene.exists():
        candidates = [p.name for p in root.iterdir() if p.is_dir()] if root.exists() else []
        raise FileNotFoundError(f"Scene not found: {scene}. Available directories: {candidates}")

    image_dirs = [
        p
        for p in scene.rglob("*")
        if p.is_dir() and (p.name.lower() in {"images", "input", "rgb"} or p.name.lower().startswith("images_"))
    ]
    print(f"Scene: {scene}")
    print(f"Candidate image dirs: {[str(p) for p in image_dirs]}")
    if not image_dirs:
        raise RuntimeError("No common image directory name found. Check the Mip-NeRF extraction layout.")


if __name__ == "__main__":
    main()
