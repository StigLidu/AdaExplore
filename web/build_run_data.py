#!/usr/bin/env python3
"""
Pack one MCTS run directory (e.g. experiments/MCTS_0120_KB-l2_50/2_56/) into a
single JSON consumed by the web visualization.

Usage:
    python web/build_run_data.py <run_dir> <output.json> [--label "Level 2 / Problem 56"]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


STEP_LOG_RE = re.compile(r"step_(\d+)_log\.json$")


def step_num(path: str | os.PathLike) -> int:
    m = STEP_LOG_RE.search(str(path))
    return int(m.group(1)) if m else -1


def short_error(metrics: dict) -> str | None:
    """Return a short, human-friendly error message if the kernel did not pass."""
    if not metrics:
        return None
    if metrics.get("compiled") and metrics.get("correctness"):
        return None
    md = metrics.get("metadata") or {}
    err_name = md.get("runtime_error_name") or md.get("compilation_error_name")
    err_text = (
        md.get("runtime_error")
        or md.get("compilation_error")
        or md.get("error")
        or ""
    )
    # Keep only the last frame of the error for readability.
    if err_text:
        err_text = err_text.strip().splitlines()
        err_text = err_text[-1] if err_text else ""
    if err_name and err_text:
        return f"{err_name}: {err_text[:200]}"
    if err_name:
        return err_name
    if err_text:
        return err_text[:240]
    if metrics.get("compiled") is False:
        return "compile failed"
    if metrics.get("correctness") is False:
        return "incorrect output"
    return None


def build(run_dir: Path, label: str, include_prompts: bool = False) -> dict:
    logs = sorted(run_dir.glob("step_*_log.json"), key=step_num)
    if not logs:
        raise SystemExit(f"No step_*_log.json under {run_dir}")

    reference_src = ""
    ref_path = run_dir / "reference_src.py"
    if ref_path.exists():
        reference_src = ref_path.read_text()

    global_best_kernel = ""
    gbk_candidates = sorted(run_dir.glob("global_best_kernel_*.py"))
    if gbk_candidates:
        global_best_kernel = gbk_candidates[-1].read_text()

    steps: list[dict] = []
    for log_path in logs:
        s = step_num(log_path)
        with log_path.open() as f:
            log = json.load(f)

        code_path = run_dir / f"step_{s}.py"
        code = code_path.read_text() if code_path.exists() else ""

        metrics_path = run_dir / f"step_{s}_metrics.json"
        metrics = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
            except json.JSONDecodeError:
                metrics = {}

        prompt = ""
        if include_prompts:
            prompt_path = run_dir / f"step_{s}_prompt.txt"
            if prompt_path.exists():
                prompt = prompt_path.read_text()

        score = log.get("score") or [0, 0, 0]
        compiled, correct, fast_p = (
            bool(score[0]),
            bool(score[1]),
            float(score[2] or 0.0),
        )

        rs = metrics.get("runtime_stats") or {}

        steps.append({
            "step": s,
            "node_id": log["node_id"],
            "parent_node_id": log.get("parent_node_id"),
            "context_node_ids": log.get("context_node_ids") or [],
            "depth": log.get("depth", 0),
            "created_by": log.get("created_by", "unknown"),
            "visits": log.get("visits", 0),
            "total_reward": log.get("total_reward", 0.0),
            "max_reward": log.get("max_reward", 0.0),
            "avg_reward": log.get("avg_reward", 0.0),
            "score": {
                "compiled": compiled,
                "correct": correct,
                "fast_p": fast_p,
            },
            "runtime_us": metrics.get("runtime"),
            "baseline_time_us": rs.get("baseline_time"),
            "hardware": rs.get("hardware") or (metrics.get("metadata") or {}).get("hardware"),
            "global_best_node_id": log.get("global_best_node_id"),
            "global_best_fast_p": (log.get("global_best_score") or [0, 0, 0])[2],
            "error": short_error(metrics),
            "code": code,
            "prompt": prompt,
        })

    return {
        "label": label,
        "run_dir": str(run_dir),
        "total_steps": len(steps) - 1,  # exclude dummy root
        "reference_src": reference_src,
        "global_best_kernel": global_best_kernel,
        "steps": steps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--label", default=None, help="Display name for this run")
    ap.add_argument("--include-prompts", action="store_true",
                    help="Embed step_N_prompt.txt files (large; default off)")
    args = ap.parse_args()

    label = args.label or args.run_dir.name
    data = build(args.run_dir, label, include_prompts=args.include_prompts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2))

    n_steps = data["total_steps"]
    final_best = data["steps"][-1]["global_best_fast_p"]
    print(f"Wrote {args.output}: {n_steps} steps, final best fast_p = {final_best:.2f}x")


if __name__ == "__main__":
    main()
