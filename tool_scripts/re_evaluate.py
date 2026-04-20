# re-evaluate the global best kernel in a log folder
import os
import sys
import argparse
import json
import re

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.utils import read_metrics, load_test_source
from src.eval import wrapped_eval_kernel_against_ref, KernelExecResult

SPEEDUP_CLAMP = (0.1, 10.0)
NO_RUNTIME = 1e9


def natural_sort_key(text):
    """Convert text to a list of strings and numbers for natural sorting."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]


def parse_args():
    parser = argparse.ArgumentParser(description="Re-evaluate the global best kernel in a log folder")
    parser.add_argument("--log_folder", type=str, required=True, help="Path to the log folder")
    parser.add_argument("--result_folders", type=str, nargs="+", default=None,
                        help="Specific result sub-folders to evaluate (e.g. 2_4 2_5). Default: all.")
    parser.add_argument("--num_correct_trials", type=int, default=5)
    parser.add_argument("--num_perf_trials", type=int, default=100)
    parser.add_argument("--measure_performance", action="store_true", default=True)
    parser.add_argument("--no_measure_performance", dest="measure_performance", action="store_false")
    parser.add_argument("--backend", type=str, default="triton", choices=["triton", "cuda"])
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--use_remote_eval", action="store_true", default=False)
    parser.add_argument("--eval_all", action="store_true", default=False)
    parser.add_argument("--remote_eval_url", type=str, default="http://127.0.0.1:12017")
    parser.add_argument("--skip_save_result", action="store_true", default=False)
    parser.add_argument("--step", type=str, default="")
    parser.add_argument("--gpu_name", type=str, default="A6000")
    parser.add_argument("--use_recorded_baseline_time", action="store_true", default=False)
    parser.add_argument("--force_re_evaluate", action="store_true", default=False)
    return parser.parse_args()


def evaluate_kernel(kernel_path: str, level: int, problem_id: int, args: argparse.Namespace) -> KernelExecResult:
    """Evaluate a single kernel file against the reference implementation."""
    with open(kernel_path, "r") as f:
        kernel = f.read()
    return wrapped_eval_kernel_against_ref(
        original_model_src=load_test_source("KB", level, problem_id)[1],
        custom_model_src=kernel,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        measure_performance=args.measure_performance,
        backend=args.backend,
        dtype_str=args.dtype,
        device=args.device,
        use_remote_eval=args.use_remote_eval,
        remote_eval_url=args.remote_eval_url,
        level = str(level) if args.use_recorded_baseline_time else None,
        problem_id = str(problem_id) if args.use_recorded_baseline_time else None,
        gpu_name = args.gpu_name if args.use_recorded_baseline_time else None,
    )


def filter_logs(logs: list[str]) -> list[str]:
    """Filter out _BUG folders when a corresponding normal folder exists."""
    all_logs = set(logs)
    filtered = []
    for log in logs:
        if log.endswith("_BUG"):
            normal_log = log.replace("_BUG", "")
            if normal_log in all_logs:
                continue
            tqdm.write(f"WARNING: Bug folder {log} has no corresponding normal folder, using it as-is")
        filtered.append(log)
    return filtered


def save_metrics(path: str, result: KernelExecResult, kernel_path: str):
    """Serialize evaluation result to a JSON file."""
    metrics = result.to_dict()
    metrics["kernel_path"] = kernel_path
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def collect_candidate_kernels(log_dir: str) -> list[str]:
    """Return sorted list of numbered kernel .py files in a log directory."""
    paths = []
    for name in sorted(os.listdir(log_dir), key=natural_sort_key):
        if name.endswith(".py") and name.split("_")[-1].split(".")[0].isdigit() and "best" not in name:
            paths.append(os.path.join(log_dir, name))
    return paths


def find_best_kernel(kernel_paths: list[str], level: int, problem_id: int,
                     args: argparse.Namespace) -> tuple[KernelExecResult, str, list[dict]]:
    """Evaluate all candidate kernels and return (best_result, best_path, step_curve).

    step_curve is a list of dicts recording the current-best runtime at each step,
    suitable for plotting optimisation progress: [{"step": int, "best_runtime": float}, ...].
    """
    best_runtime, best_path, best_result = NO_RUNTIME, None, None
    step_curve = []

    for kp in kernel_paths:
        result = evaluate_kernel(kp, level, problem_id, args)
        runtime = result.runtime_stats.get("mean", NO_RUNTIME)
        step = int(os.path.basename(kp).split("_")[-1].split(".")[0])

        if runtime < best_runtime or best_path is None:
            best_runtime = runtime
            best_path = kp
            best_result = result
            std = result.runtime_stats.get("std", NO_RUNTIME)
            fast_p = result.runtime_stats.get("fast_p", 0.0)
            tqdm.write(f"Step {step}: New best runtime: {best_runtime:.4f} (std {std:.4f}, fast_p {fast_p:.4f})")

        step_curve.append({
            "step": step,
            "current_runtime": runtime if runtime != NO_RUNTIME else None,
            "best_runtime": best_runtime if best_runtime != NO_RUNTIME else None,
            "fast_p": result.runtime_stats.get("fast_p", None) if result.runtime_stats else None,
        })

    return best_result, best_path, step_curve


def main():
    args = parse_args()
    log_folder = args.log_folder

    if not os.path.exists(log_folder):
        sys.exit(f"Error: Log folder '{log_folder}' does not exist")

    try:
        num_steps = int(log_folder.split("_")[-2].split("-")[0])
        print(f"Number of steps: {num_steps}")
    except (ValueError, IndexError):
        print("Warning: Could not extract number of steps from log folder name")

    if args.result_folders:
        logs = filter_logs([f for f in args.result_folders if os.path.isdir(os.path.join(log_folder, f))])
    else:
        logs = filter_logs(sorted(os.listdir(log_folder), key=natural_sort_key))

    speed_ups = []
    eval_all_best_speed_ups = []
    correct_count = 0

    for log in (pbar := tqdm(logs, desc="Evaluating kernels")):
        log_dir = os.path.join(log_folder, log)

        # Determine kernel path
        suffix = f"global_best_kernel_{args.step}.py" if args.step else "global_best_kernel.py"
        kernel_path = os.path.join(log_dir, suffix)
        if not os.path.exists(kernel_path):
            continue
        if os.path.exists(os.path.join(log_dir, f"global_best_metrics_on_sf_{args.step}.json")) and not args.force_re_evaluate: # already re-evaluated
            tqdm.write(f"Skipping {log} because it has already been re-evaluated")
            continue

        level, problem_id = map(int, log.split("_")[:2])

        # --- Evaluate the recorded global-best kernel ---
        result = evaluate_kernel(kernel_path, level, problem_id, args)

        if result.correctness:
            correct_count += 1

        best_speed_up = max(min(SPEEDUP_CLAMP[1], result.runtime_stats.get("fast_p", 0.0)), SPEEDUP_CLAMP[0])
        speed_ups.append(best_speed_up)

        if not args.skip_save_result:
            save_metrics(os.path.join(log_dir, f"global_best_metrics_on_sf_{args.step}.json"), result, kernel_path)

        # --- Optionally re-evaluate all candidate kernels to find the true best ---
        re_eval_path = os.path.join(log_dir, "global_best_metrics_re_evaluated.json")
        step_best_curve_path = os.path.join(log_dir, "step_best_curve.json")
        if args.eval_all and not (os.path.exists(re_eval_path) and os.path.exists(step_best_curve_path)):
            candidates = collect_candidate_kernels(log_dir)
            best_result, best_path, step_curve = find_best_kernel(candidates, level, problem_id, args)

            best_runtime = result.runtime_stats.get("mean", 0.0)
            re_best_runtime = best_result.runtime_stats.get("mean", NO_RUNTIME) if best_result else NO_RUNTIME
            tqdm.write(f"Recorded runtime: {best_runtime:.4f}, Re-evaluated best: {re_best_runtime:.4f}")

            if not args.skip_save_result:
                if best_result:
                    save_metrics(re_eval_path, best_result, best_path)
                # Save step-wise current-best curve for plotting
                with open(step_best_curve_path, "w") as f:
                    json.dump(step_curve, f, indent=2)

        # --- Collect eval-all speed-ups ---
        if args.eval_all:
            assert os.path.exists(re_eval_path), f"Re-evaluated metrics missing for {log}"
            with open(re_eval_path, "r") as f:
                re_metrics = json.load(f)
            eval_all_best_speed_ups.append(max(min(SPEEDUP_CLAMP[1], re_metrics.get("runtime_stats", {}).get("fast_p", 0.0)), SPEEDUP_CLAMP[0]))
            tqdm.write(f"{re_metrics}")

        avg_speed_up = sum(speed_ups) / len(speed_ups)
        pbar.set_postfix(avg_speed_up=f"{avg_speed_up:.4f}x", correct=correct_count, total=len(speed_ups))
        tqdm.write(f"{level}_{problem_id}: {result}")

    if speed_ups:
        avg_speed_up = sum(speed_ups) / len(speed_ups)
        print(f"Average speed up: {avg_speed_up:.4f}x, correct count: {correct_count}, total count: {len(speed_ups)}")
    if eval_all_best_speed_ups:
        print(f"Eval-all best speed up: {sum(eval_all_best_speed_ups) / len(eval_all_best_speed_ups):.4f}x")


if __name__ == "__main__":
    main()