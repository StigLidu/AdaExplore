# A loop of kernel generation --> unit test --> reviewer --> fix --> repeat

import os
import logging
import argparse
from agent.inference_server import create_inference_server, query_inference_server
from tqdm import tqdm
from src.eval import KernelExecResult
from agent.utils import extract_edits, str_replace, calculate_score, copy_step_files, load_test_source, REPO_TOP_PATH, EXAMPLE_ARCH_SRC, EXAMPLE_NEW_ARCH_SRC
from agent.actions import single_large_step, single_small_step
from agent.mcts_utils import _parse_metrics_txt
import torch

logger = logging.getLogger(__name__)


def load_from_logs(log_path: str):
    """
    Load previous kernels and metrics from log files.
    For IRS agent, large_loop_id is always 0.
    
    Args:
        log_path: Path to the log directory
    
    Returns:
        Tuple of (previous_kernels, previous_metrics, start_step, local_best_kernel, local_best_metric, local_best_score)
    """
    previous_kernels = []
    previous_metrics = []
    local_best_kernel = None
    local_best_metric = None
    local_best_score = -1.0
    
    if not os.path.exists(log_path):
        logger.warning(f"Log path {log_path} does not exist")
        return previous_kernels, previous_metrics, 0, local_best_kernel, local_best_metric, local_best_score
    
    # Find all proposal and tune files (large_loop_id is always 0 for IRS)
    step_files = []
    for f in os.listdir(log_path):
        if f.startswith("proposal_0_") and f.endswith(".py"):
            try:
                step_idx = int(f.split("_")[2].split(".")[0])
                step_files.append(("proposal", step_idx, f))
            except ValueError:
                continue
        elif f.startswith("tune_0_") and f.endswith(".py"):
            try:
                step_idx = int(f.split("_")[2].split(".")[0])
                step_files.append(("tune", step_idx, f))
            except ValueError:
                continue
    
    if not step_files:
        logger.warning(f"No step files found in {log_path}")
        return previous_kernels, previous_metrics, 0, local_best_kernel, local_best_metric, local_best_score
    
    # Sort by step index
    step_files.sort(key=lambda x: x[1])
    
    max_step = 0
    for step_type, step_idx, kernel_file in step_files:
        kernel_path = os.path.join(log_path, kernel_file)
        metrics_path = os.path.join(log_path, kernel_file.replace(".py", "_metrics.txt"))
        
        if not os.path.exists(kernel_path):
            logger.warning(f"Kernel file {kernel_path} not found, skipping")
            continue
        
        if not os.path.exists(metrics_path):
            logger.warning(f"Metrics file {metrics_path} not found, skipping")
            continue
        
        # Load kernel
        with open(kernel_path, "r") as f:
            kernel = f.read()
        
        # Load metrics
        with open(metrics_path, "r") as f:
            metrics_str = f.read()
        
        metrics = _parse_metrics_txt(metrics_str)
        
        previous_kernels.append(kernel)
        previous_metrics.append(metrics)
        
        # Update best
        score = calculate_score(metrics)
        if local_best_kernel is None or score > local_best_score:
            local_best_score = score
            local_best_kernel = kernel
            local_best_metric = metrics
        
        max_step = max(max_step, step_idx)
        logger.debug(f"Loaded {step_type} step {step_idx}")
    
    logger.info(f"Loaded {len(previous_kernels)} steps from {log_path}, max_step={max_step}, "
                f"local_best_score={local_best_score}")
    
    return previous_kernels, previous_metrics, max_step, local_best_kernel, local_best_metric, local_best_score


def process_single_problem(inference_server, args):
    """
    Process a single problem: run small loop (which will generate initial kernel internally if needed).
    
    Args:
        ref_arch_src: Reference architecture source
        inference_server: Inference server instance
        args: Arguments namespace (should have level, problem_id, save_path set)
    
    Returns:
        local_best_kernel, local_best_metrics
    """
    if args.save_path is None:
        args.save_path = os.path.join(
            REPO_TOP_PATH,
            "outputs",
            f"IRS_{args.refine_steps}_{args.level}_{args.problem_id}",
        )
    os.makedirs(args.save_path, exist_ok=True)
    
    local_best_kernel, local_best_metrics = run_small_loop(
        args.task_params["arc_src"], inference_server, None, None, args, log_path=args.save_path
    )
    
    # Save results
    with open(os.path.join(args.save_path, "local_best_kernel.py"), "w") as f:
        f.write(local_best_kernel)
    with open(os.path.join(args.save_path, "local_best_metrics.txt"), "w") as f:
        f.write(str(local_best_metrics))

    return local_best_kernel, local_best_metrics

def run_small_loop(
    ref_arch_src: str,
    inference_server,
    initial_kernel: str,
    initial_metrics: KernelExecResult,
    args: argparse.Namespace,
    large_loop_id: int = 0,
    log_path: str = None,
):
    start_step = 0
    
    # Resume from existing logs if specified (only for IRS agent)
    if args.agent_type == "IRS" and args.resume_from is not None:
        previous_kernels, previous_metrics, max_step, local_best_kernel, local_best_metric, local_best_score = load_from_logs(args.resume_from)
        loaded_count = len(previous_kernels)
        # Keep only the last max_memory_round steps to match normal flow behavior
        previous_kernels = previous_kernels[-args.max_memory_round:]
        previous_metrics = previous_metrics[-args.max_memory_round:]
        start_step = max_step
        logger.info(f"Resumed from {args.resume_from}, loaded {loaded_count} steps (keeping last {len(previous_kernels)} for memory), will continue from step {start_step + 1}")
        
        # Copy existing files to new log_path if different
        if log_path and log_path != args.resume_from:
            copy_step_files(args.resume_from, log_path)
    else:
        previous_kernels = [initial_kernel] if initial_kernel is not None else []
        previous_metrics = [initial_metrics] if initial_metrics is not None else []
        local_best_score = calculate_score(initial_metrics) if initial_metrics is not None else -1.0
        local_best_kernel = initial_kernel
        local_best_metric = initial_metrics

    # start_step is the last completed step index, so we continue from start_step
    for i in tqdm(range(start_step, args.refine_steps), desc=f"Small Loop on problem {args.level}_{args.problem_id}", initial=start_step, total=args.refine_steps):
        logger.debug(f"Running kernel {i+1} of {args.refine_steps}")

        if len(previous_kernels) == 0:
            # do a single large step to get the initial kernel and metrics
            proposal_kernel, proposal_metrics, logs = single_large_step(ref_arch_src, inference_server, [], [], args)
            previous_kernels.append(proposal_kernel)
            previous_metrics.append(proposal_metrics)
            local_best_kernel = proposal_kernel
            local_best_metric = proposal_metrics
            local_best_score = calculate_score(proposal_metrics)
            with open(os.path.join(log_path, f"proposal_{large_loop_id}_{i+1}_prompt.txt"), "w") as f:
                f.write(logs["proposer_prompt"])
            with open(os.path.join(log_path, f"proposal_{large_loop_id}_{i+1}.py"), "w") as f:
                f.write(proposal_kernel)
            with open(os.path.join(log_path, f"proposal_{large_loop_id}_{i+1}_metrics.txt"), "w") as f:
                f.write(str(proposal_metrics))
            logger.debug(f"Proposal Metrics: {proposal_metrics}")
            continue
        
        # small step
        tuned_kernel, tuned_metrics, logs = single_small_step(ref_arch_src, inference_server, previous_kernels, previous_metrics, args)

        if log_path is not None:
            with open(os.path.join(log_path, f"tune_{large_loop_id}_{i+1}_prompt.txt"), "w") as f:
                f.write(logs["tuner_prompt"])
            with open(os.path.join(log_path, f"tune_{large_loop_id}_{i+1}.py"), "w") as f:
                f.write(tuned_kernel)
            with open(os.path.join(log_path, f"tune_{large_loop_id}_{i+1}_metrics.txt"), "w") as f:
                f.write(str(tuned_metrics))

        previous_kernels.append(tuned_kernel)
        previous_metrics.append(tuned_metrics)

        # keep the previous_kernels and previous_metrics list length at most max_memory_round
        if len(previous_kernels) > args.max_memory_round:
            previous_kernels.pop(0)
            previous_metrics.pop(0)

        score = calculate_score(tuned_metrics)
        if score > local_best_score:
            local_best_score = score
            local_best_kernel = tuned_kernel
            local_best_metric = tuned_metrics

    logger.debug(f"Local Best Score: {local_best_score}, Local Best Metric: {local_best_metric}")
    return local_best_kernel, local_best_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_source", type=str, default="KB", choices=["KB"])
    # for kernelbench, we need to specify the level and problem id
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--problem_id", type=int, default=1)
    parser.add_argument("--dtype_str", type=str, default="fp32")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--server_type", type=str, default="azure", choices=["azure", "openai", "claude"])
    parser.add_argument("--model_name", type=str, default="gpt-5-mini")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_completion_tokens", type=int, default=16384)
    parser.add_argument("--test_list_path", type=str, default=None)
    parser.add_argument("--test_list_resume_idx", type=int, default=0)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--gpu_name", type=str, default="A6000")
    parser.add_argument("--gpu_architecture", type=str, default="Ampere")
    parser.add_argument("--general_memory_path", type=str, default="results/memory/general_memory_v4_200.txt")

    # for small loop
    parser.add_argument("--refine_steps", type=int, default=25)
    parser.add_argument("--max_memory_round", type=int, default=5)
    parser.add_argument("--disable_reviewer", action="store_true", default=False)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()
    args.agent_type = "IRS"

    torch.cuda.set_device(args.gpu_id)
    inference_server = create_inference_server(server_type=args.server_type)

    if args.debug:
        logger.debug("Inference Server Created, starting to test...")
        result = query_inference_server(inference_server, model_name="gpt-4o", prompt="Hello, world!")
        logger.debug(result)

    if args.test_list_path is None:
        # Single problem mode
        problem_name, ref_arch_src = load_test_source(args.test_source, args.level, args.problem_id)
        args.task_params = {
            "arc_src": ref_arch_src,
            "gpu_name": args.gpu_name,
            "gpu_architecture": args.gpu_architecture,
            "dtype_str": args.dtype_str,
            "example_arch_src": EXAMPLE_ARCH_SRC,
            "example_new_arch_src": EXAMPLE_NEW_ARCH_SRC,
        }
        process_single_problem(inference_server, args)
    else:
        # Batch mode: process multiple problems
        with open(args.test_list_path, "r") as f:
            test_list = [line.strip() for line in f.readlines()]
        
        correct_count = 0
        sum_speedup = 0
        total_count = len(test_list)
        
        for test_problem in tqdm(test_list[args.test_list_resume_idx:]):
            test_level, test_problem_id = map(int, test_problem.split())
            args.level = test_level
            args.problem_id = test_problem_id
            args.save_path = os.path.join(
                REPO_TOP_PATH,
                "outputs",
                f"IRS_{args.refine_steps}_{test_level}_{test_problem_id}",
            )
            
            problem_name, ref_arch_src = load_test_source(args.test_source, test_level, test_problem_id)
            logger.info(f"Loaded problem {problem_name} from {args.test_source} level {test_level} problem {test_problem_id}")
            
            args.task_params = {
                "arc_src": ref_arch_src,
                "gpu_name": args.gpu_name,
                "gpu_architecture": args.gpu_architecture,
                "dtype_str": args.dtype_str,
                "example_arch_src": EXAMPLE_ARCH_SRC,
                "example_new_arch_src": EXAMPLE_NEW_ARCH_SRC,
            }

            # Process this problem
            local_best_kernel, local_best_metrics = process_single_problem(inference_server, args)
            
            if local_best_metrics.correctness:
                correct_count += 1
                sum_speedup += local_best_metrics.runtime_stats["fast_p"]
        
        print(f"Total Correct: {correct_count} / {total_count} ({correct_count / total_count * 100:.2f}%)")
        print(f"Average Speedup: {sum_speedup / total_count:.2f}")