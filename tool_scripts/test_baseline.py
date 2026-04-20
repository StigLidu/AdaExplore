#!/usr/bin/env python3
"""
Test baseline performance for KernelBench problems
Tests PyTorch eager execution and optionally torch.compile with Triton backend

Usage examples:

1. Test a single problem:
   python test_baseline_perf.py --level 1 --problem_id 1 --num_trials 100

2. Generate baseline JSON for all problems (PyTorch Eager):
   python test_baseline_perf.py --generate_all --gpu_name A10G_modal --num_trials 100

3. Generate baseline JSON with torch.compile:
   python test_baseline_perf.py --generate_all --gpu_name A10G_modal \
       --use_torch_compile --torch_compile_backend inductor \
       --torch_compile_mode default --num_trials 100

4. Generate baseline JSON for specific levels:
   python test_baseline_perf.py --generate_all --gpu_name A10G_modal \
       --levels 1 2 --num_trials 100
"""

import os
import sys
import argparse
import torch
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.utils import load_test_source
from src.eval import (
    eval_kernel_against_ref,
    load_original_model_and_inputs,
    time_execution_with_cuda_event,
    get_timing_stats,
    set_seed,
)
from src.dataset import construct_problem_dataset_from_problem_dir
from src.utils import read_file

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "datasets", "KernelBench")
TIMING_DIR = os.path.join(REPO_TOP_PATH, "results", "timing")


def measure_baseline_performance(
    ref_arch_src: str,
    ref_arch_name: str,
    num_trials: int = 100,
    use_torch_compile: bool = False,
    torch_compile_backend: str = "inductor",
    torch_compile_mode: str = "default",
    device: int = 0,
    verbose: bool = False,
) -> dict:
    """
    Measure baseline performance of a reference architecture
    
    Args:
        ref_arch_src: Source code of the reference architecture
        ref_arch_name: Name of the reference architecture
        num_trials: Number of performance trials
        use_torch_compile: Whether to use torch.compile
        torch_compile_backend: Backend for torch.compile (default: "inductor" which uses Triton)
        torch_compile_mode: Mode for torch.compile (default, reduce-overhead, max-autotune, etc.)
        device: CUDA device ID
        verbose: Whether to print verbose output
    
    Returns:
        Dictionary with timing statistics
    """

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        ref_arch_src, context
    )
    
    try:
        torch.cuda.set_device(device)
        torch.cuda.synchronize(device=device)
        set_seed(42)
        inputs = get_inputs()
        set_seed(42)
        init_inputs = get_init_inputs()
        
        # Move inputs to device
        inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]
        init_inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x
            for x in init_inputs
        ]
        
        # Initialize PyTorch model
        model = Model(*init_inputs)
        
        if use_torch_compile:
            if verbose:
                print(f"Using torch.compile with backend={torch_compile_backend}, mode={torch_compile_mode}")
            model = torch.compile(
                model, 
                backend=torch_compile_backend, 
                mode=torch_compile_mode
            )
        else:
            if verbose:
                print(f"Using PyTorch Eager Execution")
        
        model = model.cuda(device=device)
        torch.cuda.synchronize(device=device)
        
        # Warm up
        for _ in range(3):
            _ = model(*inputs)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device=device)
        
        # Measure performance
        elapsed_times = time_execution_with_cuda_event(
            model, *inputs, num_trials=num_trials, verbose=verbose, device=device
        )
        runtime_stats = get_timing_stats(elapsed_times, device=device)
        
        if verbose:
            print(f"{ref_arch_name} - {runtime_stats}")
        
        return runtime_stats
            
    except Exception as e:
        print(f"[Error] Failed to measure performance for {ref_arch_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def fetch_ref_arch_from_dataset(dataset: list[str], problem_id: int) -> tuple[str, str, str]:
    """
    Fetch the reference architecture from the problem directory
    problem_id should be logical index (1-indexed), matching the problem_id in the problem_name

    Returns:
        ref_arch_path: str, the path to the reference architecture
        ref_arch_name: str, the name of the reference architecture
        ref_arch_src: str, the source code of the reference architecture
    """
    ref_arch_path = None
    
    for file in dataset:
        if file.split("/")[-1].split("_")[0] == str(problem_id):
            ref_arch_path = file
            break
    if ref_arch_path is None:
        raise ValueError(f"No reference architecture found for problem_id {problem_id}")
    
    ref_arch_src = read_file(ref_arch_path)
    ref_arch_name = ref_arch_path.split("/")[-1]
    return (ref_arch_path, ref_arch_name, ref_arch_src)


def generate_baseline_json(
    hardware_name: str,
    output_dir: str,
    num_trials: int = 100,
    use_torch_compile: bool = False,
    torch_compile_backend: str = "inductor",
    torch_compile_mode: str = "default",
    device: int = 0,
    verbose: bool = False,
    levels: list = None,
) -> dict:
    """
    Generate baseline JSON file for all KernelBench problems
    
    Args:
        hardware_name: Name of the hardware (e.g., "A10G_modal")
        output_dir: Directory to save the JSON file
        num_trials: Number of performance trials
        use_torch_compile: Whether to use torch.compile
        torch_compile_backend: Backend for torch.compile
        torch_compile_mode: Mode for torch.compile
        device: CUDA device ID
        verbose: Whether to print verbose output
        levels: List of levels to process (default: [1, 2, 3])
    
    Returns:
        Dictionary with timing results
    """
    if levels is None:
        levels = [1, 2, 3, 4]
    
    # Determine output path early so we can load existing results
    if use_torch_compile:
        if torch_compile_mode:
            filename = f"baseline_time_torch_compile_{torch_compile_backend}_{torch_compile_mode}.json"
        else:
            filename = f"baseline_time_torch_compile_{torch_compile_backend}.json"
    else:
        filename = "baseline_time_torch.json"
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    
    # Load existing results to skip already-generated tasks
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            json_results = json.load(f)
        print(f"Loaded existing results from {save_path}")
    else:
        json_results = {}
    
    device_name = torch.cuda.get_device_name(device)
    
    for level in levels:
        PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, f"level{level}")
        dataset = construct_problem_dataset_from_problem_dir(PROBLEM_DIR)
        json_results.setdefault(f"level{level}", {})
        
        num_problems = len(dataset)
        skipped = 0
        for problem_id in tqdm(range(1, num_problems + 1), desc=f"Level {level}"):
            try:
                ref_arch_path, ref_arch_name, ref_arch_src = fetch_ref_arch_from_dataset(dataset, problem_id)
                
                if ref_arch_name in json_results[f"level{level}"]:
                    skipped += 1
                    continue
                
                runtime_stats = measure_baseline_performance(
                    ref_arch_src=ref_arch_src,
                    ref_arch_name=ref_arch_name,
                    num_trials=num_trials,
                    use_torch_compile=use_torch_compile,
                    torch_compile_backend=torch_compile_backend,
                    torch_compile_mode=torch_compile_mode,
                    device=device,
                    verbose=verbose,
                )
                
                if runtime_stats is not None:
                    runtime_stats["hardware"] = device_name
                    runtime_stats["device"] = f"cuda:{device}"
                    runtime_stats["num_trials"] = num_trials
                    json_results[f"level{level}"][ref_arch_name] = runtime_stats
                else:
                    if verbose:
                        tqdm.write(f"[Warning] Failed to measure {ref_arch_name}, skipping...")
                        
            except Exception as e:
                if verbose:
                    tqdm.write(f"[Error] Failed to process {level}/{problem_id}: {e}")
                continue
        
        if skipped > 0:
            print(f"  Level {level}: skipped {skipped} already-generated tasks")
    
    with open(save_path, "w") as f:
        json.dump(json_results, f, indent=4)
    
    print(f"\nResults saved to {save_path}")
    return json_results


def parse_args():
    parser = argparse.ArgumentParser(description="Test baseline performance for KernelBench problems")
    parser.add_argument("--level", type=int, default=None, help="Problem level (e.g., 2). Required if not using --generate_all")
    parser.add_argument("--problem_id", type=int, default=None, help="Problem ID (1-indexed). Required if not using --generate_all")
    parser.add_argument("--test_source", type=str, default="KB", choices=["KB", "SYN"], help="Test source: KB (KernelBench) or SYN (Synthesized)")
    parser.add_argument("--num_trials", type=int, default=100, help="Number of performance trials")
    parser.add_argument("--device", type=int, default=3, help="CUDA device ID")
    parser.add_argument("--use_torch_compile", action="store_true", help="Use torch.compile (default backend is inductor/Triton)")
    parser.add_argument("--torch_compile_backend", type=str, default="inductor", help="Backend for torch.compile (default: inductor)")
    parser.add_argument("--torch_compile_mode", type=str, default="default", 
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                        help="Mode for torch.compile")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path (optional, for single problem mode)")
    
    # Batch generation options
    parser.add_argument("--generate_all", action="store_true", help="Generate baseline JSON for all problems (batch mode)")
    parser.add_argument("--gpu_name", type=str, default=None, help="GPU name for batch mode (e.g., 'A10G_modal')")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for batch mode (default: results/timing/{gpu_name})")
    parser.add_argument("--levels", type=int, nargs="+", default=None, help="Levels to process in batch mode (default: [1, 2, 3, 4])")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        sys.exit(1)

    if args.device >= torch.cuda.device_count():
        print(f"Error: Device {args.device} not available. Only {torch.cuda.device_count()} device(s) available.")
        sys.exit(1)
    
    # Batch generation mode
    if args.generate_all:
        if args.gpu_name is None:
            print("Error: --gpu_name is required when using --generate_all")
            sys.exit(1)
        
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = os.path.join(TIMING_DIR, args.gpu_name)
        
        print(f"Generating baseline JSON for all KernelBench problems")
        print(f"GPU: {args.gpu_name}")
        print(f"Output directory: {output_dir}")
        print(f"Number of trials: {args.num_trials}")
        
        config_name = "torch.compile" if args.use_torch_compile else "PyTorch Eager"
        if args.use_torch_compile:
            config_name += f" ({args.torch_compile_backend}, {args.torch_compile_mode})"
        print(f"Configuration: {config_name}")
        print("-" * 60)
        
        levels = args.levels if args.levels else [1, 2, 3, 4]
        generate_baseline_json(
            hardware_name=args.gpu_name,
            output_dir=output_dir,
            num_trials=args.num_trials,
            use_torch_compile=args.use_torch_compile,
            torch_compile_backend=args.torch_compile_backend,
            torch_compile_mode=args.torch_compile_mode,
            device=args.device,    
            verbose=args.verbose,
            levels=levels,
        )
        return
    
    # Single problem mode (original behavior)
    if args.level is None or args.problem_id is None:
        print("Error: --level and --problem_id are required when not using --generate_all")
        sys.exit(1)
    
    print(f"Testing baseline performance on {torch.cuda.get_device_name(args.device)}")
    print(f"Level: {args.level}, Problem ID: {args.problem_id}")
    print(f"Test Source: {args.test_source}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Device: {args.device}")
    print("-" * 60)
    
    # Load test source
    try:
        problem_name, ref_arch_src = load_test_source(args.test_source, args.level, args.problem_id)
        print(f"Problem: {problem_name}")
    except Exception as e:
        print(f"Error loading test source: {e}")
        sys.exit(1)
    
    # Measure baseline performance
    config_name = "torch.compile" if args.use_torch_compile else "PyTorch Eager"
    if args.use_torch_compile:
        config_name += f" ({args.torch_compile_backend}, {args.torch_compile_mode})"
    
    print(f"\nMeasuring baseline with {config_name}...")
    runtime_stats = measure_baseline_performance(
        ref_arch_src=ref_arch_src,
        ref_arch_name=problem_name,
        num_trials=args.num_trials,
        use_torch_compile=args.use_torch_compile,
        torch_compile_backend=args.torch_compile_backend,
        torch_compile_mode=args.torch_compile_mode,
        device=args.device,
        verbose=args.verbose,
    )

    runtime_stats_again = eval_kernel_against_ref(
        original_model_src=ref_arch_src,
        custom_model_src=ref_arch_src.replace("Model", "ModelNew"),
        num_perf_trials=args.num_trials,
        measure_performance=True,
        device=args.device,
    )

    print(f"Runtime stats: {runtime_stats}")
    print(f"Runtime stats again: {runtime_stats_again}")
    
    if runtime_stats is None:
        print("Failed to measure baseline performance")
        sys.exit(1)
    
    # Print results
    print("\n" + "=" * 60)
    print("Baseline Performance Results:")
    print("-" * 60)
    print(f"Mean: {runtime_stats.get('mean', 'N/A'):.4f} ms")
    print(f"Std:  {runtime_stats.get('std', 'N/A'):.4f} ms")
    print(f"Min:  {runtime_stats.get('min', 'N/A'):.4f} ms")
    print(f"Max:  {runtime_stats.get('max', 'N/A'):.4f} ms")
    print("=" * 60)
    
    # Save to file if requested
    if args.output:
        result = {
            "problem_name": problem_name,
            "level": args.level,
            "problem_id": args.problem_id,
            "test_source": args.test_source,
            "config": {
                "use_torch_compile": args.use_torch_compile,
                "torch_compile_backend": args.torch_compile_backend if args.use_torch_compile else None,
                "torch_compile_mode": args.torch_compile_mode if args.use_torch_compile else None,
                "num_trials": args.num_trials,
                "device": args.device,
                "device_name": torch.cuda.get_device_name(args.device),
            },
            "runtime_stats": runtime_stats,
        }
        
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

