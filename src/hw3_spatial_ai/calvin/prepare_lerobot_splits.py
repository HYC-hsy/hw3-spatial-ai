from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ENV_TOKENS = {
    "A": ("env_a", "environment_a", "calvin_a", "task_A", "task_a", "A"),
    "B": ("env_b", "environment_b", "calvin_b", "task_B", "task_b", "B"),
    "C": ("env_c", "environment_c", "calvin_c", "task_C", "task_c", "C"),
    "D": ("env_d", "environment_d", "calvin_d", "task_D", "task_d", "D"),
}


def infer_env(path: Path) -> str | None:
    parts = [p.lower() for p in path.parts]
    joined = "/".join(parts)
    for env, tokens in ENV_TOKENS.items():
        for token in tokens:
            t = token.lower()
            if t in parts or t in joined:
                return env
    return None


def make_link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def collect_env_roots(data_root: Path) -> dict[str, list[Path]]:
    candidates: dict[str, list[Path]] = {k: [] for k in ENV_TOKENS}
    for path in data_root.iterdir():
        env = infer_env(path)
        if env:
            candidates[env].append(path)
    if any(candidates.values()):
        return candidates

    for path in data_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".parquet", ".arrow", ".hdf5", ".h5", ".npz"}:
            env = infer_env(path)
            if env:
                candidates[env].append(path)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare CALVIN splits for LeRobot experiments. The script preserves the original files "
            "with symlinks by default. If the dataset is already in LeRobot format, it creates split "
            "directories that LeRobot can read or that can be adapted with minimal path changes."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlinking.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    env_roots = collect_env_roots(data_root)
    manifest = {env: [str(p) for p in paths] for env, paths in env_roots.items()}

    splits = {
        "env_a_train": ["A"],
        "env_abc_train": ["A", "B", "C"],
        "env_d_eval": ["D"],
    }

    for split, envs in splits.items():
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for env in envs:
            paths = env_roots.get(env, [])
            if not paths:
                print(f"Warning: no files/directories inferred for env {env}.")
            for src in paths:
                dst = split_dir / f"env_{env.lower()}" / src.name
                make_link_or_copy(src, dst, args.copy)

    with (output_root / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote split directories to {output_root}")
    print(f"Wrote manifest to {output_root / 'split_manifest.json'}")
    print("If LeRobot cannot load these directories directly, use this manifest to finish the dataset-specific conversion.")


if __name__ == "__main__":
    main()

