"""Evaluate a single kernel file against its reference implementation."""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.utils import load_test_source
from src.eval import wrapped_eval_kernel_against_ref


def main():
    parser = argparse.ArgumentParser(description="Evaluate one kernel against reference")
    parser.add_argument("kernel_path", type=str, help="Path to the kernel .py file")
    parser.add_argument("--test_source", type=str, default="KB", choices=["KB", "SYN", "FIT", "MLSYS"])
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--problem_id", type=str, required=True)
    parser.add_argument("--num_correct_trials", type=int, default=5)
    parser.add_argument("--num_perf_trials", type=int, default=200)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--backend", type=str, default="triton", choices=["triton", "cuda"])
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--use_remote_eval", action="store_true", default=False)
    parser.add_argument("--remote_eval_url", type=str, default="http://0.0.0.0:12017")
    args = parser.parse_args()

    with open(args.kernel_path, "r") as f:
        kernel_src = f.read()

    ref_src = load_test_source(args.test_source, args.level, args.problem_id)[1]

    result = wrapped_eval_kernel_against_ref(
        original_model_src=ref_src,
        custom_model_src=kernel_src,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        measure_performance=True,
        backend=args.backend,
        dtype_str=args.dtype,
        device=args.device,
        use_remote_eval=args.use_remote_eval,
        remote_eval_url=args.remote_eval_url,
        test_source=args.test_source,
        level=args.level,
        problem_id=args.problem_id,
    )

    print(result)

    print(f"Compiled:    {result.compiled}")
    print(f"Correctness: {result.correctness}")
    if result.runtime_stats:
        print(f"fast_p:      {result.runtime_stats.get('fast_p', 'N/A')}")
        print(f"mean (us):   {result.runtime_stats.get('mean', 'N/A')}")
        print(f"std (us):    {result.runtime_stats.get('std', 'N/A')}")
    else:
        print("No runtime stats (kernel did not pass correctness)")


if __name__ == "__main__":
    main()
