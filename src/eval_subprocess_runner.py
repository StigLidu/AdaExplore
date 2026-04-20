"""
Subprocess runner for eval_kernel_against_ref
This script is executed in a separate process to isolate runtime errors
"""
import sys
import os
import json
import torch
import logging

# Configure logging to output to stderr to avoid mixing with JSON output in stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Explicitly set to stderr
)

# Get the parent directory (repo root) so we can import from src
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == 'src':
    src_dir = os.path.dirname(script_dir)
else:
    src_dir = script_dir

# Add the parent directory to the path so we can import from src
sys.path.insert(0, src_dir)

from src.eval import eval_kernel_against_ref, KernelExecResult

# Parse arguments from JSON
args = json.loads(sys.argv[1])
original_model_src = args['original_model_src']
custom_model_src = args['custom_model_src']
seed_num = args['seed_num']
num_correct_trials = args['num_correct_trials']
num_perf_trials = args['num_perf_trials']
verbose = args['verbose']
measure_performance = args['measure_performance']
build_dir = args.get('build_dir')
device_val = args['device']
backend = args['backend']
dtype_str = args['dtype_str']
gpu_name = args['gpu_name']
level = args['level']
problem_id = args['problem_id']

# Set device
if torch.cuda.is_available():
    torch.cuda.set_device(device_val)
    device = device_val
else:
    device = None

try:
    result = eval_kernel_against_ref(
        original_model_src=original_model_src,
        custom_model_src=custom_model_src,
        seed_num=seed_num,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        verbose=verbose,
        measure_performance=measure_performance,
        build_dir=build_dir,
        device=device,
        backend=backend,
        dtype_str=dtype_str,
        gpu_name=gpu_name,
        level=level,
        problem_id=problem_id,
    )
    
    # Serialize result to JSON
    if result is None:
        result_dict = None
    else:
        result_dict = {
            'compiled': result.compiled,
            'correctness': result.correctness,
            'metadata': result.metadata,
            'runtime': result.runtime,
            'runtime_stats': result.runtime_stats,
        }
    
    print(json.dumps(result_dict))
    sys.exit(0)
except Exception as e:
    import traceback
    error_dict = {
        'error': str(e),
        'traceback': traceback.format_exc(),
        'error_type': type(e).__name__
    }
    print(json.dumps(error_dict))
    sys.exit(1)

