import os
import time
import logging
import json
from agent.inference_server import create_inference_server, query_inference_server
from tqdm import tqdm
import argparse
from agent.actions import single_large_step
from agent.small_loop import run_small_loop
from agent.utils import calculate_score, copy_step_files, load_test_source, read_metrics, REPO_TOP_PATH
import torch
import numpy as np

logger = logging.getLogger(__name__)

def run_large_loop(ref_arch_src: str, inference_server: str, args: argparse.Namespace, log_path: str = None):
    # previously experienced kernels and metrics
    kernel_pool = []
    metrics_pool = []
    proposal_ids = []  # Track original proposal IDs for context logging
    elite_pool = []
    elite_metrics_pool = []
    elite_proposal_ids = []

    resume_start = 0
    if args.agent_type == "PS" and args.resume_from is not None:
        resume_dir = args.resume_from
        for pid in range(1, args.proposal_steps + 1):
            kernel_path = os.path.join(resume_dir, f"proposal_{pid}.py")
            if not os.path.exists(kernel_path):
                break
            metrics_path = next(
                (os.path.join(resume_dir, f"proposal_{pid}_metrics{ext}")
                 for ext in [".json", ".txt"]
                 if os.path.exists(os.path.join(resume_dir, f"proposal_{pid}_metrics{ext}"))),
                None,
            )
            if metrics_path is None:
                break
            with open(kernel_path, "r") as f:
                kernel_src = f.read()
            metrics = read_metrics(metrics_path, full_metrics=True)
            kernel_pool.append(kernel_src)
            metrics_pool.append(metrics)
            proposal_ids.append(pid)

        if kernel_pool:
            sorted_data = sorted(
                zip(kernel_pool, metrics_pool, proposal_ids),
                key=lambda x: calculate_score(x[1]),
                reverse=True,
            )
            elite_pool, elite_metrics_pool, elite_proposal_ids = [list(x) for x in zip(*sorted_data)]

        resume_start = len(kernel_pool)
        logger.info(f"Resumed {resume_start} proposals from {resume_dir}")
        if log_path and log_path != args.resume_from:
            copy_step_files(args.resume_from, log_path)

    def _first_k_idx(n: int, k: int) -> list[int]:
        if k <= 0 or n <= 0:
            return []
        return list(range(0, min(k, n)))

    def _last_k_idx(n: int, k: int) -> list[int]:
        if k <= 0 or n <= 0:
            return []
        return list(range(max(0, n - k), n))

    for i in tqdm(range(resume_start, args.proposal_steps), desc=f"Large Loop on problem {args.level}_{args.problem_id}"):
        logger.debug(f"Running proposal {i+1} of {args.proposal_steps}")
        
        # Provide both "recent" (trajectory) and "elite" pools to the proposer.
        recent_context_ids: list[int] = []
        elite_context_ids: list[int] = []
        recent_idx: list[int] = []
        elite_idx: list[int] = []
        pool_size = int(getattr(args, "pool_size", 0) or 0)

        if args.agent_type == "IRL":
            recent_idx = _last_k_idx(len(proposal_ids), pool_size)
            recent_context_ids = [proposal_ids[j] for j in recent_idx]
        elif args.agent_type in ["IRB", "PS"]:
            elite_idx = _first_k_idx(len(elite_proposal_ids), pool_size)
            elite_context_ids = [elite_proposal_ids[j] for j in elite_idx]
        elif args.agent_type == "IRLE":
            k_recent = pool_size // 2
            k_elite = pool_size - k_recent

            recent_idx = _last_k_idx(len(proposal_ids), k_recent)
            recent_context_ids = [proposal_ids[j] for j in recent_idx]

            # Sample elite (correct-only) by softmax over fast_p
            if k_elite > 0 and elite_metrics_pool:
                candidate_idx: list[int] = []
                candidate_fast_p: list[float] = []
                for j, m in enumerate(elite_metrics_pool):
                    score = calculate_score(m)
                    if score[0] == 1 and score[1] == 1 and elite_proposal_ids[j] not in recent_context_ids:
                        candidate_idx.append(j)
                        candidate_fast_p.append(score[2])

                if len(candidate_idx) > 0:
                    choose_n = min(k_elite, len(candidate_idx))
                    tau = float(getattr(args, "softmax_temperature", 1.0))
                    tau = max(1e-6, tau)
                    x = np.asarray(candidate_fast_p, dtype=np.float64) / tau
                    x = x - np.max(x)
                    p = np.exp(x)
                    p = p / np.sum(p)
                    picked_local = np.random.choice(
                        len(candidate_idx), size=choose_n, replace=False, p=p
                    )
                    picked_idx = [candidate_idx[t] for t in picked_local.tolist()]
                    elite_context_ids = [elite_proposal_ids[j] for j in picked_idx]
                    elite_idx = picked_idx
        else:
            raise ValueError(f"Invalid agent type: {args.agent_type}")

        recent_kernels_to_pass = [kernel_pool[j] for j in recent_idx] if recent_idx else []
        recent_metrics_to_pass = [metrics_pool[j] for j in recent_idx] if recent_idx else []
        elite_kernels_to_pass = [elite_pool[j] for j in elite_idx] if elite_idx else []
        elite_metrics_to_pass = [elite_metrics_pool[j] for j in elite_idx] if elite_idx else []

        proposal_kernel, proposal_metrics, logs = single_large_step(
            ref_arch_src,
            inference_server,
            recent_kernels_to_pass,
            recent_metrics_to_pass,
            args,
            context_ids=recent_context_ids,
            elite_kernel_pool=elite_kernels_to_pass,
            elite_metrics_pool=elite_metrics_to_pass,
            elite_context_ids=elite_context_ids,
        )
        
        # log the proposal
        if log_path is not None:
            with open(os.path.join(log_path, f"proposal_{i+1}.txt"), "w") as f:
                f.write(logs["proposer_prompt"])
            with open(os.path.join(log_path, f"proposal_{i+1}.py"), "w") as f:
                f.write(logs["proposal_kernel"])
            with open(os.path.join(log_path, f"proposal_{i+1}_metrics.txt"), "w") as f:
                f.write(str(logs["proposal_metrics"]))
            
            # Log step info with context (similar to MCTS tree_structure.txt)
            step_log = {
                "step": i + 1,
                "step_type": "large_step",
                "context": {
                    "recent": recent_context_ids,
                    "elite": elite_context_ids,
                },
                "context_metrics": {
                    "recent": [str(m) for m in recent_metrics_to_pass],
                    "elite": [str(m) for m in elite_metrics_to_pass],
                },
                "context_size": {
                    "recent": len(recent_context_ids),
                    "elite": len(elite_context_ids),
                },
                "compiled": proposal_metrics.compiled if proposal_metrics else False,
                "correctness": proposal_metrics.correctness if proposal_metrics else False,
                "runtime": proposal_metrics.runtime if proposal_metrics else -1.0,
                "score": calculate_score(proposal_metrics),
            }
            with open(os.path.join(log_path, f"proposal_{i+1}_log.json"), "w") as f:
                json.dump(step_log, f, indent=2)

        if args.refine_steps > 0:
            local_best_kernel, local_best_metrics = run_small_loop(
                ref_arch_src, 
                inference_server, 
                proposal_kernel, 
                proposal_metrics,
                args,
                large_loop_id=i+1,
                log_path=log_path,
            )
        else:
            local_best_kernel = proposal_kernel
            local_best_metrics = proposal_metrics

        logger.debug(f"Local Best Metrics: {local_best_metrics}")
        logger.debug(f"Local Best Score: {calculate_score(local_best_metrics)}")
        kernel_pool.append(local_best_kernel)
        metrics_pool.append(local_best_metrics)
        proposal_ids.append(i + 1)  # Track the original proposal ID
        
        # Maintain a separately sorted elite pool for best-kernel context (do NOT reorder kernel_pool).
        sorted_data = sorted(
            zip(kernel_pool, metrics_pool, proposal_ids),
            key=lambda x: calculate_score(x[1]),
            reverse=True,
        )
        elite_pool, elite_metrics_pool, elite_proposal_ids = zip(*sorted_data)
        elite_pool = list(elite_pool)
        elite_metrics_pool = list(elite_metrics_pool)
        elite_proposal_ids = list(elite_proposal_ids)

    if len(elite_pool) == 0:
        raise ValueError("No kernels were generated; empty pool.")
    # Elite pool is already sorted by score (best first).

    return elite_pool[0], elite_metrics_pool[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_source", type=str, default="KB", choices=["KB"])
    # for kernelbench, we need to specify the level and problem id
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--problem_id", type=int, default=1)
    parser.add_argument("--test_list_path", type=str, default=None)
    parser.add_argument("--test_list_resume_idx", type=int, default=0)
    parser.add_argument("--dtype_str", type=str, default="fp32")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--server_type", type=str, default="azure", choices=["azure", "openai", "claude"])
    parser.add_argument("--model_name", type=str, default="gpt-5-mini")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_completion_tokens", type=int, default=16384)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--gpu_name", type=str, default="A6000")
    parser.add_argument("--gpu_architecture", type=str, default="Ampere")
    parser.add_argument("--general_memory_path", type=str, default="agentprompt/general_memory_v0.txt")

    # for large loop
    parser.add_argument("--proposal_steps", type=int, default=4)
    parser.add_argument("--refine_steps", type=int, default=4) # if 0, no small loop will be used
    parser.add_argument("--pool_size", type=int, default=5) # if 0, no pool will be used, degrade to parallel search
    parser.add_argument("--disable_reviewer", action="store_true", default=False)

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)
    start_time = time.strftime("%m%d-%H%M%S", time.localtime())
    if args.refine_steps == 0 and args.pool_size == 0:
        exp_prefix = "PS"
    else:
        exp_prefix = f"FIXLS"

    problem_name, ref_arch_src = load_test_source(args.test_source, args.level, args.problem_id)

    if args.debug:
        logger.debug("-" * 10 + "Reference Architecture" + "-" * 10)
        logger.debug(ref_arch_src)
        logger.debug("-" * 10 + "End of Reference Architecture" + "-" * 10)

    inference_server = create_inference_server(
        server_type=args.server_type
    )

    if args.debug:
        logger.debug("Inference Server Created, starting to test...")
        result = query_inference_server(inference_server, model_name="gpt-4o", prompt="Hello, world!")
        logger.debug(result)

    if args.test_list_path is None:
        # test single problem
        if args.save_path is None:
            args.save_path = os.path.join(REPO_TOP_PATH, "outputs", f"{exp_prefix}_{start_time}_{args.proposal_steps}_{args.refine_steps}_{args.level}_{args.problem_id}")
        os.makedirs(args.save_path, exist_ok=True)
        global_best_kernel, global_best_metrics = run_large_loop(ref_arch_src, inference_server, args)
        logger.info(f"Global Best Kernel: {global_best_kernel}")
        exit()

    # test list of problems
    with open(args.test_list_path, "r") as f:
        test_list = f.readlines()
    test_list = [line.strip() for line in test_list]
    correct_count = 0
    sum_speedup = 0
    total_count = len(test_list)
    for test_problem in tqdm(test_list[args.test_list_resume_idx:]):
        test_level, test_problem_id = int(test_problem.split(" ")[0].strip()), int(test_problem.split(" ")[1].strip())
        logger.info(f"Testing problem {test_problem} at {test_level} {test_problem_id}")
        args.save_path = os.path.join(REPO_TOP_PATH, "outputs", f"{exp_prefix}_{start_time}_{args.proposal_steps}_{args.refine_steps}_{test_level}_{test_problem_id}")
        os.makedirs(args.save_path, exist_ok=True)
        problem_name, ref_arch_src = load_test_source(args.test_source, test_level, test_problem_id)
        global_best_kernel, global_best_metrics = run_large_loop(ref_arch_src, inference_server, args)
        if global_best_metrics.correctness:
            correct_count += 1
            sum_speedup += global_best_metrics.runtime_stats["fast_p"]
    logger.info(f"Total Correct: {correct_count} / {total_count} ({correct_count / total_count * 100:.2f}%)")
    logger.info(f"Average Speedup: {sum_speedup / total_count:.2f}")