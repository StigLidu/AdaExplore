import os
import argparse
import time
import shutil
import yaml
import torch
from tqdm import tqdm
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import logging

REPO_TOP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_TOP_PATH)

# Configure logging based on DEBUG mode
# DEBUG mode is controlled by args.debug, but we need to set up logging before args is parsed
# So we check environment variable or will update later
DEBUG_MODE = os.environ.get("DEBUG", "false").lower() == "true"
LOG_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Suppress httpx INFO logs (HTTP request logs)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
from agent.large_loop import run_large_loop
from agent.small_loop import run_small_loop
from agent.inference_server import create_inference_server
from agent.utils import read_metrics
from agent.utils import load_test_source, calculate_score, load_config_from_yaml, load_tasks_from_test_list, EXAMPLE_ARCH_SRC, EXAMPLE_NEW_ARCH_SRC
from skill_memory.skill_memory import update_memory
from agent.mcts import mcts_search

def agent_entry(args: argparse.Namespace, inference_server, level, problem_id):
    """
    Entry point for the agent.
    
    For KB/SYN: level and problem_id are integers
    For FIT/MLSYS: level is op_type name (str), problem_id is problem name (str)
    For TBG: level is ignored, problem_id is 1-indexed id or task filename
    """
    result_save_path = os.path.join(args.save_path, f"{level}_{problem_id}")
    os.makedirs(result_save_path, exist_ok=True)
    problem_name, ref_arch_src = load_test_source(args.test_source, level, problem_id, suffix=getattr(args, 'suffix', ''))

    args.level = level
    args.problem_id = problem_id
    if not hasattr(args, "gpu_id"):
        # Default to the first GPU id when a specific assignment was not injected
        args.gpu_id = args.gpu_ids[0] if getattr(args, "gpu_ids", []) else 0
    
    # Set task_params for proposer prompt
    if args.test_source == "KB":
        args.task_params = {
            "arc_src": ref_arch_src,
            "gpu_name": args.gpu_name,
            "gpu_architecture": args.gpu_architecture,
            "dtype_str": args.dtype_str,
            "example_arch_src": EXAMPLE_ARCH_SRC,
            "example_new_arch_src": EXAMPLE_NEW_ARCH_SRC,
        }
    elif args.test_source in ("FIT", "MLSYS"):
        args.task_params = {
            "definition": json.dumps(ref_arch_src, indent=4),
            "target_gpu": args.gpu_name,
            "dtype_str": str(ref_arch_src.get("inputs", "unknown")),
            "arc_src": json.dumps(ref_arch_src, indent=4),
        }
    elif args.test_source == "TBG":
        args.task_params = {
            "task_id": str(problem_id),
            "target_gpu": args.gpu_name,
            "arc_src": ref_arch_src,
            "dtype_str": args.dtype_str,
            "gpu_name": args.gpu_name,
            "gpu_architecture": args.gpu_architecture,
        }
    elif args.test_source == "SYN":
        args.task_params = {
            "arc_src": ref_arch_src,
            "gpu_name": args.gpu_name,
            "gpu_architecture": args.gpu_architecture,
            "dtype_str": args.dtype_str,
            "example_arch_src": EXAMPLE_ARCH_SRC,
            "example_new_arch_src": EXAMPLE_NEW_ARCH_SRC,
        }
    # save the reference architecture source code
    with open(os.path.join(result_save_path, f"reference_src.py"), "w") as f:
        if args.test_source in ("FIT", "MLSYS"):
            # For FIT/MLSYS, ref_arch_src is a dict, convert to JSON string
            f.write(json.dumps(ref_arch_src, indent=4))
        else:
            # For KB/SYN, ref_arch_src is already a string (Python code)
            f.write(ref_arch_src)

    if args.agent_type == "IRS":
        # Iterative refinement with small loop
        args.refine_steps = args.total_steps
        global_best_kernel, global_best_metrics = run_small_loop(
            ref_arch_src, inference_server, None, None, args, log_path=result_save_path
        )
    elif args.agent_type in ["IRL", "IRB", "IRLE"]:
        # Iterative refinement with large loop
        # - IRL: iterative refinement with large loop (recent context)
        # - IRB: iterative refinement with best previous kernels as context (elite context)
        # - IRLE: iterative refinement with large step + evolution (recent + elite sampled by softmax)
        args.proposal_steps = args.total_steps
        args.refine_steps = 0
        global_best_kernel, global_best_metrics = run_large_loop(ref_arch_src, inference_server, args, log_path=result_save_path)
    elif args.agent_type == "PS":
        # Parallel search with no iterative refinement
        args.proposal_steps = args.total_steps
        args.refine_steps = 0
        args.pool_size = 0
        global_best_kernel, global_best_metrics = run_large_loop(ref_arch_src, inference_server, args, log_path=result_save_path)
    elif args.agent_type == "MCTS":
        # MCTS search
        global_best_kernel, global_best_metrics = mcts_search(
            ref_arch_src, inference_server, args, log_path=result_save_path,
        )
    else:
        raise ValueError(f"Invalid agent type: {args.agent_type}")

    # save the global best kernel and metrics
    with open(os.path.join(result_save_path, f"global_best_kernel_{args.total_steps}.py"), "w") as f:
        f.write(global_best_kernel)
    with open(os.path.join(result_save_path, f"global_best_metrics_{args.total_steps}.json"), "w") as f:
        json.dump(global_best_metrics.to_dict(), f, indent=4)

    if args.memory_update:
        lock_file_path = args.general_memory_path + ".lock"
        lock_timeout = 1800  # 30 minutes timeout
        lock_acquired = False
        start_time = time.time()
        with open(lock_file_path, "w") as lock_file:
            while time.time() - start_time < lock_timeout:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    lock_acquired = True
                    break
                except IOError:
                    time.sleep(0.1)  # Wait 100ms before retrying
            
            if not lock_acquired:
                raise TimeoutError(f"Failed to acquire lock for {lock_file_path} within {lock_timeout} seconds")
            
            try:
                update_memory(memory_path=args.general_memory_path, log_path=result_save_path, server=inference_server, model_name=args.model_name, filter_max_difference=True)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return global_best_kernel, global_best_metrics

def worker_process(task_info, args_dict, gpu_id):
    """
    Worker function for multiprocessing.
    Each process handles one task and creates its own inference_server.
    """
    # Ensure sys.path is set correctly in subprocess
    REPO_TOP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if REPO_TOP_PATH not in sys.path:
        sys.path.append(REPO_TOP_PATH)
    
    # Reconstruct args from dict
    args = argparse.Namespace(**args_dict)
    args.level = task_info['level']
    args.problem_id = task_info['problem_id']

    # Append level_problem_id to resume_from base path and check if it exists
    if args.resume_from:
        resume_path = os.path.join(args.resume_from, f"{args.level}_{args.problem_id}")
        if os.path.exists(resume_path):
            args.resume_from = resume_path
        else:
            args.resume_from = None

    # Check if the result already exists
    result_save_path = os.path.join(args.save_path, f"{task_info['level']}_{task_info['problem_id']}")
    if os.path.exists(result_save_path):
        files = os.listdir(result_save_path)
        kernel_file = f"global_best_kernel_{args.total_steps}.py"
        metrics_json = f"global_best_metrics_{args.total_steps}.json"
        metrics_txt = f"global_best_metrics_{args.total_steps}.txt"
        
        # Find metrics file (prefer JSON over TXT)
        metrics_file = None
        if kernel_file in files:
            if metrics_json in files:
                metrics_file = os.path.join(result_save_path, metrics_json)
            elif metrics_txt in files:
                metrics_file = os.path.join(result_save_path, metrics_txt)
        
        if metrics_file:
            correctness, fast_p = read_metrics(metrics_file)
            return {
                'level': task_info['level'],
                'problem_id': task_info['problem_id'],
                'success': True,
                'correctness': correctness,
                'speedup': fast_p,
                'kernel': None,
                'metrics': None
            }
        else:
            if not (args.resume_from is not None and args.resume_from == result_save_path):
                shutil.rmtree(result_save_path)

    # Create inference server for this process
    inference_server = create_inference_server(server_type=args.server_type)

    # Run agent_entry
    # Only set CUDA device if not using remote evaluation (remote eval doesn't need local CUDA)
    if not args_dict.get('use_remote_eval', False):
        torch.cuda.set_device(gpu_id)
    args.gpu_id = gpu_id
    try:
        global_best_kernel, global_best_metrics = agent_entry(
            args, inference_server, task_info['level'], task_info['problem_id']
        )
        return {
            'level': task_info['level'],
            'problem_id': task_info['problem_id'],
            'success': True,
            'correctness': global_best_metrics.correctness,
            'speedup': global_best_metrics.runtime_stats.get("fast_p", 0.0) if global_best_metrics.correctness else 0.0,
            'kernel': global_best_kernel,
            'metrics': global_best_metrics
        }
    except Exception as e:
        print(f"Error processing level {task_info['level']} problem {task_info['problem_id']} on GPU {gpu_id}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'level': task_info['level'],
            'problem_id': task_info['problem_id'],
            'success': False,
            'correctness': False,
            'speedup': 0.0,
            'error': str(e)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Test Configs
    parser.add_argument("--test_source", type=str, default="KB", choices=["KB", "SYN", "FIT", "MLSYS", "TBG"])
    parser.add_argument("--suffix", type=str, default="", help="Dataset folder suffix, e.g. '_v1' for KernelBench_v1")
    parser.add_argument("--agent_type", type=str, default="MCTS", choices=["IRS", "IRL", "IRLE", "PS", "MCTS"])
    # for kernelbench (KB): level is int (1, 2, 3), problem_id is int (1-indexed)
    # for synthesized_data (SYN): level is int (version), problem_id is int (1-indexed)
    # for flashinfer_trace (FIT): level is op_type name (e.g., "gemm"), problem_id is problem name (e.g., "gemm_n128_k2048")
    # for mlsys26-contest (MLSYS): same as FIT, but dataset root differs
    # for TritonBench-G (TBG): level is ignored; problem_id is 1-indexed id or task filename
    parser.add_argument("--level", type=str, default="2")
    parser.add_argument("--problem_id", type=str, default="1")
    parser.add_argument("--test_list_path", type=str, default=None)
    parser.add_argument("--dtype_str", type=str, default="fp32")
    parser.add_argument("--gpu_name", type=str, default="A6000")
    parser.add_argument("--gpu_architecture", type=str, default="Ampere")
    # Default to the first visible GPU; override for multi-GPU or remote setups.
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0])
    parser.add_argument("--num_processes", type=int, default=8, help="Number of processes to use")
    parser.add_argument("--use_remote_eval", action="store_true", default=True, help="Use remote evaluation service")
    parser.add_argument("--remote_eval_url", type=str, default="http://127.0.0.1:12017", help="URL for remote evaluation service")
    #TODO: combine remote eval and gpu_ids

    # Base Model Configs
    parser.add_argument("--server_type", type=str, default="azure", choices=["azure", "openai", "claude"])
    parser.add_argument("--model_name", type=str, default="gpt-5-mini")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_completion_tokens", type=int, default=16384)
    
    # Knowledge Base Configs
    parser.add_argument(
        "--general_memory_path",
        type=str,
        default=None,
    )
    parser.add_argument("--memory_update", action="store_true", default=False)
    parser.add_argument("--knowledge_1_threshold", type=int, default=3) # The threshold for knowledge_1

    # Resume Configs
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from existing log folder (e.g., outputs/MCTS_xxx)")
    # [TODO]: currently only support MCTS and IRS agent

    # Shared Search Configs
    parser.add_argument("--total_steps", type=int, default=25)
    parser.add_argument("--max_memory_round", type=int, default=5)
    parser.add_argument("--pool_size", type=int, default=5)
    parser.add_argument("--disable_reviewer", action="store_true", default=False)
    parser.add_argument("--force_reviser", action="store_true", default=False) # Whether to force reviser agent to run even if all previous kernels are wrong

    # Search Configs for MCTS
    parser.add_argument("--exploration_weight", type=float, default=0.25)  # UCB1 exploration constant
    parser.add_argument("--expand_exploration_ratio", type=float, default=1.0)  # Scale factor for expand-action exploration
    parser.add_argument("--reward_alpha", type=float, default=1.0)  # α*max + (1-α)*avg in UCB1 (1.0=max, 0.0=avg)
    parser.add_argument("--small_step_limit", type=int, default=2)  # Max number of small steps per node
    parser.add_argument("--p_large", type=float, default=0.25)  # Probability of large step (for MCTS)
    parser.add_argument("--pool_size_extra_max", type=int, default=3)  # Max extra nodes from other components
    parser.add_argument("--softmax_temperature", type=float, default=1.0)  # Temperature for softmax sampling
    parser.add_argument("--geometric_p", type=float, default=0.5)  # Parameter for truncated geometric distribution (higher = fewer extra nodes)
    parser.add_argument("--dummy", action="store_true", default=False)  # Whether to use dummy actions for debugging

    # Logging Configs
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    
    # Config file
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file to load parameters from")

    args = parser.parse_args()
    
    # Load config from YAML file if provided
    args = load_config_from_yaml(args, parser)
    assert args.num_processes % len(args.gpu_ids) == 0, "Number of processes must be a multiple of the number of GPUs"

    start_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    try:
        memory_version = args.general_memory_path.split("_")[-1].split(".")[0]
        memory_version = "v" + memory_version if not memory_version.startswith("v") else memory_version
    except:
        memory_version = "v0"

    inference_server = create_inference_server(server_type=args.server_type)
    
    if args.save_path is None:
        args.save_path = os.path.join(REPO_TOP_PATH, "outputs", \
            f"{args.agent_type}_{memory_version}_{args.test_source}_{args.total_steps}_{start_time}")
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f, sort_keys=False)

    if args.test_list_path is None:
        # test on a single problem
        args.gpu_id = args.gpu_ids[0]
        global_best_kernel, global_best_metrics = agent_entry(args, inference_server, args.level, args.problem_id)
        print(f"Global Best Kernel: {global_best_kernel}")
        print(f"Global Best Metrics: {global_best_metrics}")
        print(f"Global Best Score: {calculate_score(global_best_metrics)}")
        exit()

    # Load tasks from test list file
    tasks = load_tasks_from_test_list(args.test_list_path, args.test_source)
    
    # Convert args to dict for serialization (exclude non-serializable objects)
    args_dict = vars(args).copy()
    
    # Multi-process testing with ProcessPoolExecutor
    # Set start method: 'spawn' for CUDA compatibility, 'fork' for remote_eval (faster)
    if args.use_remote_eval:
        # When using remote evaluation, worker processes don't need CUDA, so 'fork' is safe and faster
        try:
            multiprocessing.set_start_method('fork', force=True)
        except RuntimeError:
            # Start method already set, ignore
            pass
    else:
        # When using local CUDA evaluation, need 'spawn' to avoid CUDA context inheritance issues
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # Start method already set, ignore
            pass

    num_gpus = len(args.gpu_ids)
    print(f"Starting multi-process testing with {args.num_processes} processes on GPUs {args.gpu_ids}")
    results = []

    correct_count = 0
    sum_speedup = 0
    total_count = len(tasks)

    # Use appropriate context: 'fork' for remote_eval (faster), 'spawn' for local CUDA
    mp_context = multiprocessing.get_context('fork' if args.use_remote_eval else 'spawn')
    with ProcessPoolExecutor(max_workers=args.num_processes, mp_context=mp_context) as executor:
        # Submit all tasks, assigning GPU IDs in round-robin fashion
        future_to_task = {}
        for idx, task in enumerate(tasks):
            gpu_id = args.gpu_ids[idx % num_gpus]  # Round-robin GPU assignment
            future = executor.submit(worker_process, task, args_dict, gpu_id)
            future_to_task[future] = task
        
        # Process completed tasks as they finish
        with tqdm(total=total_count, desc="Processing problems") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        if result['correctness']:
                            correct_count += 1
                            sum_speedup += result['speedup']
                        print(f"Completed: level {result['level']} problem {result['problem_id']} - "
                              f"Correct: {result['correctness']}, Speedup: {result['speedup']:.4f}")
                    else:
                        print(f"Failed: level {result['level']} problem {result['problem_id']} - {result.get('error', 'Unknown error')}")
                    
                    pbar.update(1)
                except Exception as e:
                    print(f"Exception for level {task['level']} problem {task['problem_id']}: {e}")
                    pbar.update(1)
    
    print(f"Correct count: {correct_count}, Sum speedup: {sum_speedup}, Total count: {total_count}")
    if total_count > 0:
        print(f"Average speedup: {sum_speedup / total_count}")
        print(f"Accuracy: {correct_count / total_count}")
    else:
        print("Average speedup: N/A (no test problems)")
        print("Accuracy: N/A (no test problems)")
