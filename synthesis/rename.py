from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def natural_sort_key(s: str):
    # Split by digit runs so "file_2.py" < "file_10.py"
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def _resolve_repo_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _safe_rmtree(path: Path) -> None:
    if not path.exists():
        return
    shutil.rmtree(path)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def copy_and_renumber_py_files(
    *,
    source_path: Path,
    data_path: Path,
    force: bool,
    start_index: int,
) -> None:
    if not source_path.exists() or not source_path.is_dir():
        raise SystemExit(f"Source path does not exist or is not a directory: {source_path}")

    if data_path.exists():
        if not force:
            raise SystemExit(
                f"Destination already exists: {data_path}\n"
                f"Re-run with --force to overwrite it."
            )
        _safe_rmtree(data_path)

    print(f"Copying from {source_path} -> {data_path}")
    shutil.copytree(source_path, data_path)

    entries = sorted((p for p in data_path.iterdir()), key=lambda p: natural_sort_key(p.name))
    py_files = [p for p in entries if p.is_file() and p.suffix == ".py"]
    non_py = [p for p in entries if p not in py_files]

    # Remove non-.py artifacts (e.g., prompt/response .txt files) first.
    for p in non_py:
        print(f"Removing {p.name}")
        _remove_path(p)

    # Single-phase rename (no staging). We fail fast on name collisions.
    for i, src in enumerate(py_files):
        final_name = f"{start_index + i}.py"
        dst = data_path / final_name
        if src == dst:
            continue
        if dst.exists():
            raise SystemExit(
                f"Rename collision: target already exists: {dst}\n"
                f"This usually happens if the source already contains numeric filenames "
                f"like '{final_name}'. Please use a source folder without numeric names."
            )
        print(f"Renaming {src.name} -> {final_name}")
        shutil.move(str(src), str(dst))
    print(f"Done. Wrote {len(py_files)} files into {data_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a generated-data directory into a dataset folder, "
            "remove non-.py files, and renumber .py files sequentially."
        )
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="outputs/data_generation/generated_data_composite_example",
        help="Directory produced by synthesis/generate_data.py (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/KernelBench_syn/syn_v4",
        help="Destination dataset directory (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite destination if it already exists.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="First index for renamed .py files (default: 1).",
    )
    args = parser.parse_args()

    source_path = _resolve_repo_path(args.source_path)
    data_path = _resolve_repo_path(args.data_path)

    copy_and_renumber_py_files(
        source_path=source_path,
        data_path=data_path,
        force=bool(args.force),
        start_index=int(args.start_index),
    )


if __name__ == "__main__":
    main()