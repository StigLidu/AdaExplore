"""
Helpers for Evaluations
"""

import importlib
import json
import logging
import os, subprocess
import random
import shutil
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Union
import time
import uuid

import numpy as np
import requests
import torch
import torch.nn as nn
import traceback

logger = logging.getLogger(__name__)

from .format import KernelExecResult, REPO_TOP_PATH


def get_error_name(e: Exception) -> str:

    return f"{e.__class__.__module__}.{e.__class__.__name__}"


def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)

def load_original_model_and_inputs(
    model_original_src: str, context: dict
) -> tuple[nn.Module, callable, callable]:
    """
    Load class from original NN.module pytorch code
    this is pytorch reference and we feed that to model to see if there will be any improvement
    """

    try:
        compile(model_original_src, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None

    try:
        exec(model_original_src, context)  # expose to current namespace
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None

    # these should be defined in the original model code and present in the context
    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get("Model")
    return (Model, get_init_inputs_fn, get_inputs_fn)


def load_original_model_and_inputs_from_file(
    original_model_src_path: str | os.PathLike, context: dict
) -> tuple[nn.Module, callable, callable]:
    """
    Load original model + input functions from a python source file.

    Compared to `load_original_model_and_inputs`, this uses the real file path as the
    code object's filename and registers it in `linecache` so `inspect.getsource()`
    can reliably retrieve the source.
    """
    path = Path(original_model_src_path)
    if not path.exists():
        print(f"Original model file at {path} does not exist.")
        return None

    model_original_src = path.read_text()
    try:
        import linecache

        code_obj = compile(model_original_src, str(path), "exec")
        linecache.cache[str(path)] = (
            len(model_original_src),
            None,
            [line + "\n" for line in model_original_src.splitlines()],
            str(path),
        )
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None

    try:
        exec(code_obj, context)
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None

    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get("Model")

    return (Model, get_init_inputs_fn, get_inputs_fn)


def load_custom_model_with_tempfile(model_custom_src, entry_point="ModelNew"):
    """
    Writes the provided Python code string to a temporary .py file,
    dynamically imports the module so we can access the modified model class.

    Returns both a Model class and the temporary file. The temporary file must be
    deleted manually be the caller.

    This is a hack that is needed for triton code as compile / exec do not play well
    with the @triton.jit decorator.
    """

    # Create a temporary named file with a .py extension
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        # Write the code string into the file
        tmp_file.write(model_custom_src)
        # Capture the path to the file
        tempfile_path = tmp_file.name
        temp_file = tmp_file

    # Create a module specification pointing to our temp file
    spec = importlib.util.spec_from_file_location("temp_module", tempfile_path)
    # Create a new module based on that spec
    temp_module = importlib.util.module_from_spec(spec)
    # Execute the code in the module's namespace
    spec.loader.exec_module(temp_module)

    ModelNew = getattr(temp_module, entry_point)

    # Return the object (class, function, etc.) that was defined in the code
    return ModelNew, temp_file


def load_custom_model_with_tempfile_from_file(
    model_custom_src_path: str | os.PathLike, entry_point: str = "ModelNew"
) -> tuple[object, None]:
    """
    Load Triton-based custom model from a python source file.

    This mirrors `load_custom_model_with_tempfile`, but avoids writing another temp file.
    It imports directly from the provided file path.

    Returns (entry, None) so callers can share the same cleanup code path (tempfile optional).
    """
    path = Path(model_custom_src_path)
    if not path.exists():
        raise FileNotFoundError(f"Custom model file at {path} does not exist.")

    module_name = f"kb_custom_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, entry_point):
        raise AttributeError(f"Entry point '{entry_point}' not found in {path}")
    return getattr(module, entry_point), None


def load_custom_model_from_file(
    model_custom_src_path: str | os.PathLike,
    context: dict,
    build_directory: str = None,
    entry_point: str = "ModelNew",
):
    """
    Load non-Triton custom model code from a python source file via compile/exec.

    Uses the real file path as filename and registers to `linecache` so that
    `inspect.getsource()` can retrieve the source if needed.
    """
    path = Path(model_custom_src_path)
    if not path.exists():
        raise FileNotFoundError(f"Custom model file at {path} does not exist.")

    model_custom_src = path.read_text()
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        model_custom_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
        ) + model_custom_src

    import linecache

    code_obj = compile(model_custom_src, str(path), "exec")
    linecache.cache[str(path)] = (
        len(model_custom_src),
        None,
        [line + "\n" for line in model_custom_src.splitlines()],
        str(path),
    )
    exec(code_obj, context)

    return context.get(entry_point)


def load_custom_model(
    model_custom_src: str, context: dict, build_directory: str = None
) -> nn.Module:
    """
    Load class from custom NN.module pytorch code
    this is the code output by LLM with calls to custom cuda kernels
    """
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        # Add import at the start of the source code
        model_custom_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
        ) + model_custom_src

    try:
        compile(model_custom_src, "<string>", "exec")
        exec(model_custom_src, context)
        # DANGER: need to delete refernece from global namespace
    except SyntaxError as e:
        print(f"Syntax Error in custom generated code or Compilation Error {e}")
        return None

    ModelNew = context.get("ModelNew")
    return ModelNew


def _cleanup_cuda_extensions():
    """Helper function to cleanup compiled CUDA extensions"""
    # SIMON NOTE: is this necessary?
    import shutil

    torch_extensions_path = os.path.join(
        os.path.expanduser("~"), ".cache", "torch_extensions"
    )
    if os.path.exists(torch_extensions_path):
        shutil.rmtree(torch_extensions_path)


def graceful_eval_cleanup(
    curr_context: dict,
    device: torch.device,
    tempfile: tempfile.NamedTemporaryFile = None,
):
    """
    Clean up env, gpu cache, and compiled CUDA extensions after evaluation
    """  # delete ran-specific function definitions before next eval run
    del curr_context
    # Clear CUDA cache and reset GPU state
    try:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

            # does this help?
            torch.cuda.reset_peak_memory_stats(device=device)

            torch.cuda.synchronize(
                device=device
            )  # Wait for all CUDA operations to complete
    except Exception as e:
        torch.cuda.reset() # [TODO]: what is the best way to handle this? Will it cause any issues?]
    if tempfile:
        tempfile.close()
        os.remove(tempfile.name)

    # _cleanup_cuda_extensions() # SIMON NOTE: is this necessary?


def build_compile_cache(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None,
) -> tuple[bool, str, str]:
    """
    Try to build the compiled cuda code for sample and store in the cache directory
    Should be able to run on CPUs to do this massively in parallel

    Don't limit ninja to set default number of workers, let it use all the cpu cores possible
    # try do this with a subprocess
    NOTE: currently stdout_buffer does not capture all the compiler warning and failure messages
    Returns:
        tuple[bool, str]: whether compilation is successful, stdout content as string
    """
    context = {}
    stdout_buffer = StringIO()

    if verbose:
        print("[Compilation] Pre-compile custom cuda binaries")

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # sys.stdout.flush()

        # Capture stdout during compilation
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            load_custom_model(custom_model_src, context, build_dir)
            # sys.stdout.flush()

        if verbose:
            print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
    except Exception as e:
        print(
            f"[Compilation] Failed to compile custom CUDA kernel. Unable to cache, \nError: {e}"
        )
        return False, stdout_buffer.getvalue(), str(e)

    return True, stdout_buffer.getvalue(), None


def build_compile_cache_with_capturing(
    custom_model_src: str, verbose: bool = False, build_dir: os.PathLike = None
) -> tuple[int, str, str]:
    """
    Write a temporary python file to compile the custom model on CPU
    Captures the return code, stdout, and stderr
    This works for capturing, build_compile_cache does not
    """
    if build_dir:
        # Add import at the start of the source code
        custom_model_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n"
        ) + custom_model_src

    kernel_hash = hash(custom_model_src)
    # tmp is a temp python file we write to for compilation
    tmp = os.path.join(build_dir, f"tmp_{kernel_hash}.py")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)

    with open(tmp, "w", encoding="utf-8") as f:
        f.write(custom_model_src)

    # Execute the temporary Python file and capture output
    process = subprocess.Popen(
        ["python", tmp], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    returncode = process.returncode

    # Clean up temporary file
    os.remove(tmp)

    if verbose:
        print("[CPU Precompile] return code: ", returncode)
        print("[CPU Precompile] stdout: \n", stdout.decode("utf-8"))
        print("[CPU Precompile] stderr: \n", stderr.decode("utf-8"))

    return returncode, stdout.decode("utf-8"), stderr.decode("utf-8")


def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    verbose: bool = False,
    measure_performance: bool = False,
    build_dir: os.PathLike = None,
    device: Union[torch.device, int] = (
        torch.cuda.current_device() if torch.cuda.is_available() else None
    ),  # have to run on GPU
    backend: str = "cuda",  # can be 'cuda' or 'triton'
    dtype_str: str = "fp32",
    level: str = None,
    problem_id: str = None,
    gpu_name: str = None,
) -> KernelExecResult:
    """
    Evaluate the custom kernel against the original model

    num_correct_trials: number of trials to initialize different random inputs; correctness pass only if all trials pass
    num_perf_trials: run the evalutation many times to take the average
    device: GPU (cuda) device to run the evalutation on
    backend: str, either 'cuda' or 'triton', determines which backend implementation to use
    """
    # TODO: check device is busy
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,  # Decimal places
        threshold=10,  # Total number of elements before truncating
        edgeitems=3,  # Number of elements at beginning and end of dimensions
        linewidth=80,  # Maximum width before wrapping
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    baseline_time = None
    if gpu_name and problem_id is not None and level is not None:
        try:
            # load baseline time from results/timing/{gpu_name}/baseline_time_torch.json
            baseline_time_filepath = os.path.join(REPO_TOP_PATH, "results", "timing", gpu_name, f"baseline_time_torch.json")
            if not os.path.exists(baseline_time_filepath):
                if verbose:
                    print(f"[Warning] Baseline time file not found: {baseline_time_filepath}")
            else:
                with open(baseline_time_filepath, "r") as f:
                    baseline_time_json = json.load(f)
                # Convert level and problem_id to string for consistent matching
                level_key = f"level{level}"
                problem_id_str = str(problem_id)
                if level_key in baseline_time_json:
                    # baseline_time_json[level_key] is a dict with problem filenames as keys
                    for problem_name, problem_data in baseline_time_json[level_key].items():
                        # problem_name format: "1_Square_matrix_multiplication_.py"
                        # Check if it starts with "{problem_id}_"
                        if problem_name.startswith(f"{problem_id_str}_"):
                            # problem_data is a dict with "mean", "std", etc.
                            baseline_time = problem_data
                            break
        except Exception as e:
            if verbose:
                print(f"[Warning] Failed to load baseline time: {e}")
            pass
    
    # set CUDA device
    torch.cuda.set_device(device)
    is_triton = backend == "triton"
    metadata = {}  # for storing result metadata
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)  # for debugging

    if is_triton:
        # need to set env var for triton code to guarentee no wrong device shennanignas
        if isinstance(device, int):
            device_num = device
        elif isinstance(device, torch.device):
            assert (
                device.type == "cuda"
            ), "CUDA is not availible on device, cannot run Eval"
            device_num = device.index
        else:
            raise ValueError(
                f"device must be an int or torch.device, got {type(device)}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    context = {}

    if verbose:
        print(f"[Eval] Start Evalulation! on device: {device}")
        print("[Eval] Loading Original Model")

    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        original_model_src, context
    )
    set_seed(seed_num)  # set seed for reproducible input
    init_inputs = get_init_inputs()
    init_inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]

    with torch.no_grad():
        set_seed(seed_num)  # set seed for reproducible weights
        original_model = Model(*init_inputs)
        assert hasattr(original_model, "forward")
        if verbose:
            print("[Eval] Original Model Loaded")
    if verbose:
        print("[Eval] Loading and Compiling New Model with Custom CUDA Kernel")

    # this is where compilation happens
    # STAGE 1: Compile the custom model
    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        tempfile = None
        # add hash for later to distinguish between multi-turn kernels
        if is_triton:
            ModelNew, tempfile = load_custom_model_with_tempfile(
                custom_model_src, entry_point="ModelNew"
            )
        else:
            ModelNew = load_custom_model(custom_model_src, context, build_dir)
        torch.cuda.synchronize(device=device)  # not sure if this is too much
    except Exception as e:
        #print(
        #    f"Failed to compile custom CUDA kernel: Record as compilation failure. \nError: {e}"
        #)
        # TODO: add metadata for compilation error (how to we get the compilation error message?)

        if "lock" in str(e) or "No such file or directory" in str(e):
            # this is a lock file error, likely due to concurrent compilation
            # this does not necessarily mean the compilation failed, but we should retry
            print(
                f"[Eval] Lock file error during compilation, Please retry. Error: {e}"
            )
            graceful_eval_cleanup(context, device, tempfile)
            return None
        else:
            full_error = traceback.format_exc()
            metadata["compilation_error_name"] = get_error_name(e)
            metadata["compilation_error"] = full_error
            graceful_eval_cleanup(context, device, tempfile)
            return KernelExecResult(
                compiled=False, metadata=metadata
            )  # skip further steps

    # at this point we passed compilation
    # STAGE 2: Load the custom model
    try:
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            custom_model = ModelNew(*init_inputs)
            assert hasattr(custom_model, "forward")
            torch.cuda.synchronize(device=device)
        if verbose:
            print("[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        #print(
        #    f"Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. \nError: {e}"
        #)
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        graceful_eval_cleanup(context, device, tempfile)
        full_error = traceback.format_exc()
        metadata["runtime_error"] = full_error
        metadata["runtime_error_name"] = get_error_name(e)
        return KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )  # skip further steps

    kernel_exec_result = None

    # STAGE 3: Check Correctness
    if verbose:
        print("[Eval] Checking Correctness")
    try:
        kernel_exec_result = run_and_check_correctness(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=seed_num,
            device=device,
            dtype_str=dtype_str,
        )
    except Exception as e:
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        full_error = traceback.format_exc()
        metadata["runtime_error"] = full_error
        metadata["runtime_error_name"] = get_error_name(e)
        kernel_exec_result = KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )

    # STAGE 4: Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
    if measure_performance:
        try:
            if kernel_exec_result and kernel_exec_result.correctness:
                if verbose:
                    print("[Eval] Measuring Performance as Sample is Correct")

                # Clear GPU cache and synchronize to ensure consistent state before measurement
                # This is critical for performance measurement consistency across multiple runs
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(device=device)
                
                set_seed(seed_num)
                inputs = get_inputs()
                inputs = [
                    x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]
                model_new = custom_model.cuda(device=device)
                torch.cuda.synchronize(device=device)

                elapsed_times = time_execution_with_cuda_event(
                    model_new,
                    *inputs,
                    num_trials=num_perf_trials,
                    verbose=verbose,
                    device=device,
                )

                # Measure original model time
                # Clear GPU state before measuring original model to ensure fair comparison
                if baseline_time is not None:
                    mean_orig = baseline_time["mean"]
                    if verbose:
                        print(f"[Info] Using baseline time from file: {baseline_time}")
                else:
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize(device=device)
                    
                    model_ori = original_model.cuda(device=device)
                    model_ori_elapsed_times = time_execution_with_cuda_event(
                        model_ori,
                        *inputs,
                        num_trials=num_perf_trials,
                        verbose=verbose,
                        device=device,
                    )
                    mean_orig = get_timing_stats(model_ori_elapsed_times, device=device)["mean"]

                runtime_stats = get_timing_stats(elapsed_times, device=device)
                mean_new = runtime_stats["mean"]
                runtime_stats["fast_p"] = mean_orig / mean_new
                if baseline_time is not None:
                    runtime_stats["baseline_time"] = baseline_time["mean"]
                    runtime_stats["baseline_type"] = "recorded"
                else:
                    runtime_stats["baseline_type"] = "measured"
                    runtime_stats["baseline_time"] = mean_orig

                if verbose:
                    print(f"[Eval] Performance Stats: {runtime_stats}")
                kernel_exec_result.runtime = runtime_stats["mean"]
                kernel_exec_result.runtime_stats = runtime_stats
        except Exception as e:
            # TODO: seems like this is not working as expected
            # TODO: Randomness or other issues?
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            full_error = traceback.format_exc()
            metadata["runtime_error"] = full_error
            metadata["runtime_error_name"] = get_error_name(e)
            kernel_exec_result = KernelExecResult(
                compiled=True, correctness=False, metadata=metadata
            )

    graceful_eval_cleanup(context, device, tempfile)
    return kernel_exec_result

def extract_last_error(error_str: str) -> str:
    """
    Extract the last error from the error string
    """
    last_file_line = "  File " + error_str.split("\n  File ")[-1]
    # remove the first line
    last_file_line = "\n".join(last_file_line.split("\n")[1:])
    return last_file_line

def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate=True,
    max_length=1024,
):
    """
    max_length characters

    NOTE: I can't get torch truncate to work during exception handling so I have this for now
    """
    # Truncate exception message if too long
    exception_str = extract_last_error(str(exception_msg))
    if truncate and len(exception_str) > max_length:
        exception_str = exception_str[: max_length - 3] + "..."

    if verbose:
        print(f"[Exception {exception_type}] {exception_str} ")
    metadata[exception_type] = exception_str

    return metadata


def time_execution_with_cuda_event(
    kernel_fn: callable,
    *args,
    num_warmup: int = 10,  # Increased from 3 to 10 for better consistency
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
    outlier_remove_ratio: float = 0.1,
) -> list[float]:
    """
    Time a CUDA kernel function over multiple trials using torch.cuda.Event

    Args:
        kernel_fn: Function to time
        *args: Arguments to pass to kernel_fn
        num_trials: Number of timing trials to run
        verbose: Whether to print per-trial timing info
        device: CUDA device to use, if None, use current device

    Returns:
        List of elapsed times in milliseconds
    """
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()

    # Warm ups
    for _ in range(num_warmup):
        kernel_fn(*args)
        torch.cuda.synchronize(device=device)

    if verbose:
        print(
            f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}"
        )
    elapsed_times = []

    # Actual trials
    for trial in range(num_trials):
        # create event marker default is not interprocess
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        # synchronize to ensure the device is ready
        torch.cuda.synchronize(device=device)
        start_event.record()
        kernel_fn(*args)
        end_event.record()
        # Synchronize to ensure the events have completed
        torch.cuda.synchronize(device=device)

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)
        if verbose:
            print(f"Trial {trial + 1}: {elapsed_time_ms:.3g} ms")
        elapsed_times.append(elapsed_time_ms)

    # Remove outliers: drop the largest and smallest (outlier_remove_ratio/2 each) from both ends
    n = len(elapsed_times)
    if outlier_remove_ratio > 0 and n > 0:
        sorted_times = sorted(elapsed_times)
        remove_n = int(n * outlier_remove_ratio / 2)
        if remove_n > 0 and 2 * remove_n < n:
            trimmed_times = sorted_times[remove_n:-remove_n]
            if verbose:
                print(f"[Outlier Removal] Dropping {remove_n} min and {remove_n} max outliers ({n - len(trimmed_times)} in total)")
            elapsed_times = trimmed_times

    return elapsed_times


def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose=False,
    dtype_str: str = "fp32",
    seed=42,
    device=None,
) -> KernelExecResult:
    """
    run the model and check correctness,
    assume model already loaded and compiled (loaded and compiled in the caller)
    this is all on GPU, requiring cuda device and transfer .cuda()

    num_correct_trials: run the evalutation multiple times with (ideally) different random inputs to ensure correctness
    """
    pass_count = 0

    # Generate num_correct_trials seeds deterministically from the initial seed
    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]

    with torch.no_grad():

        for trial in range(num_correct_trials):

            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Eval] Generating Random Input with seed {trial_seed}")

            set_seed(trial_seed)
            inputs = get_inputs_fn()
            if dtype_str == "fp16":
                inputs = [x.cuda(device=device).to(dtype=torch.float16) if isinstance(x, torch.Tensor) else x for x in inputs]
            elif dtype_str == "fp32":
                inputs = [x.cuda(device=device).to(dtype=torch.float32) if isinstance(x, torch.Tensor) else x for x in inputs]
            elif dtype_str == "bf16":
                inputs = [x.cuda(device=device).to(dtype=torch.bfloat16) if isinstance(x, torch.Tensor) else x for x in inputs]
            else:
                raise ValueError(f"Invalid data type: {dtype_str}")

            set_seed(trial_seed)
            model = original_model_instance.cuda(device=device)

            set_seed(trial_seed)
            model_new = new_model_instance.cuda(device=device)

            output = model(*inputs)
            torch.cuda.synchronize(device=device)
            # ensure all GPU operations are completed before checking results

            try:
                output_new = model_new(*inputs)
                torch.cuda.synchronize(device=device)
                if output.shape != output_new.shape:
                    metadata = register_and_format_exception(
                        "correctness_issue",
                        f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}",
                        metadata,
                    )
                    metadata["correctness_issue_name"] = "correctness_issue"
                    if verbose:
                        print(
                            f"[FAIL] trial {trial}: Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
                        )
                    return KernelExecResult(
                        compiled=True, correctness=False, metadata=metadata
                    )

                # check output value difference
                if not torch.allclose(
                    output, output_new, atol=5e-02, rtol=5e-02
                ):  # fail
                    max_diff = torch.max(torch.abs(output - output_new)).item()
                    avg_diff = torch.mean(torch.abs(output - output_new)).item()
                    metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                    metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                    metadata["correctness_issue"] = "Output mismatch"
                    if verbose:
                        print(f"[FAIL] trial {trial}: Output mismatch")
                else:  # pass
                    pass_count += 1
                    if verbose:
                        print(f"[PASS] trial {trial}: New Model matches Model")

            except Exception as e:
                import traceback
                full_error = traceback.format_exc()
                #print("[Error] Exception happens during correctness check")
                #print(f"Error in launching kernel for ModelNew: {full_error}")

                metadata = register_and_format_exception(
                    "runtime_error", full_error, metadata, truncate=False
                )
                metadata["runtime_error_name"] = get_error_name(e)
                return KernelExecResult(
                    compiled=True, correctness=False, metadata=metadata
                )
                # break

    if verbose:
        print(
            f"[Eval] Pass count: {pass_count}, num_correct_trials: {num_correct_trials}"
        )

    # put all the useful info here!
    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    if pass_count == num_correct_trials:
        return KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    else:
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)


def check_metadata_serializable(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings
    """
    try:
        json.dumps(metadata)
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings
        metadata = {
            "eval_0": {
                k: (
                    str(v)
                    if not isinstance(
                        v, (dict, list, str, int, float, bool, type(None))
                    )
                    else v
                )
                for k, v in metadata["eval_0"].items()
            }
        }
        print(
            f"[WARNING] Metadata now converted to string: {metadata} to be JSON serializable"
        )

    return metadata


def check_metadata_serializable_all_types(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings recursively
    """

    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    try:
        json.dumps(metadata)
        return metadata
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings recursively
        converted_metadata = convert_to_serializable(metadata)
        print(
            f"[WARNING] Metadata now converted to be JSON serializable: {converted_metadata}"
        )
        return converted_metadata


################################################################################
# Performance Eval
################################################################################


def fetch_baseline_time(
    level_name: str, problem_id: int, dataset: list[str], baseline_time_filepath: str
) -> dict:
    """
    Fetch the baseline time from the time
    """
    if not os.path.exists(baseline_time_filepath):
        raise FileNotFoundError(
            f"Baseline time file not found at {baseline_time_filepath}"
        )

    with open(baseline_time_filepath, "r") as f:
        baseline_json = json.load(f)

    problem_name = dataset[problem_id].split("/")[-1]
    baseline_time = baseline_json[level_name].get(problem_name, None)
    return baseline_time


def get_timing_stats(elapsed_times: list[float], device: torch.device = None) -> dict:
    """Get timing statistics from a list of elapsed times.

    Args:
        elapsed_times: List of elapsed times in milliseconds
        device: CUDA device, record device info
    Returns:
        Dict containing mean, std, min, max and num_trials
        all timing are in ms
    """

    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    if device is not None:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)  # for debugging

    return stats

def _normalize_device(device: Union[torch.device, int]) -> int:
    """Convert device to int."""
    if isinstance(device, torch.device):
        return device.index if device.index is not None else 0
    elif device is None:
        return torch.cuda.current_device() if torch.cuda.is_available() else 0
    else:
        return int(device)


def _remote_eval_with_retry(
    url: str,
    json_data: dict,
    timeout: int,
    eval_type: str = "normal",
    max_retries: int = 3,
    initial_retry_delay: int = 2,
    raise_on_failure: bool = False
) -> KernelExecResult:
    """
    Execute remote evaluation with retry logic.
    
    Args:
        url: Endpoint URL
        json_data: Request JSON data
        timeout: Request timeout in seconds
        eval_type: "FIT", "TBG", or "normal" (for logging)
        max_retries: Maximum number of retry attempts
        initial_retry_delay: Initial delay between retries (seconds)
        raise_on_failure: If True, raise exception on failure; otherwise return error KernelExecResult
    
    Returns:
        KernelExecResult with evaluation results or error information
    """
    retry_delay = initial_retry_delay
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=json_data, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            
            # Convert response to KernelExecResult format
            if eval_type == "FIT":
                return KernelExecResult(
                    compiled=result["compiled"],
                    correctness=result["correctness"],
                    runtime=result["runtime"],
                    runtime_stats=result["runtime_stats"],
                    metadata=result["metadata"]
                )
            else:
                return KernelExecResult(**result)
                
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(f"{eval_type} evaluation timeout (attempt {attempt + 1}/{max_retries}): {e}, timeout: {timeout}")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            
            error_msg = f"{eval_type} evaluation failed after {max_retries} attempts due to timeout"
            if raise_on_failure:
                raise RuntimeError(error_msg) from e
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={'error': error_msg, 'timeout': timeout, 'attempts': max_retries}
            )
            
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(f"{eval_type} evaluation request failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            
            error_msg = f"{eval_type} evaluation failed after {max_retries} attempts: {str(last_error)}"
            if raise_on_failure:
                raise RuntimeError(error_msg) from e
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={'error': error_msg, 'attempts': max_retries}
            )
            
        except json.JSONDecodeError as e:
            error_msg = f'Failed to parse {eval_type} evaluation response'
            if raise_on_failure:
                raise RuntimeError(error_msg) from e
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={
                    'error': error_msg,
                    'json_error': str(e),
                    'response_text': response.text[:500] if 'response' in locals() else 'N/A'
                }
            )
        except Exception as e:
            error_msg = f"Unexpected error during {eval_type} evaluation: {e}"
            if raise_on_failure:
                raise RuntimeError(error_msg) from e
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={'error': error_msg, 'error_type': type(e).__name__}
            )
    
    # Should not reach here, but handle just in case
    error_msg = f"{eval_type} evaluation failed after {max_retries} attempts"
    if raise_on_failure:
        raise RuntimeError(error_msg)
    return KernelExecResult(
        compiled=False,
        correctness=False,
        metadata={'error': error_msg, 'attempts': max_retries}
    )


def _local_subprocess_eval(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int,
    num_correct_trials: int,
    num_perf_trials: int,
    verbose: bool,
    measure_performance: bool,
    build_dir: os.PathLike,
    device: int,
    backend: str,
    dtype_str: str,
    timeout: int,
    gpu_name: str = None,
    level: str = None,
    problem_id: str = None,
) -> KernelExecResult:
    """Execute evaluation in local subprocess."""
    eval_file_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(eval_file_dir) == 'src':
        src_dir = os.path.dirname(eval_file_dir)
    else:
        src_dir = eval_file_dir
    
    script_path = os.path.join(eval_file_dir, 'eval_subprocess_runner.py')
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Subprocess runner script not found at {script_path}")
    
    args_dict = {
        'original_model_src': original_model_src,
        'custom_model_src': custom_model_src,
        'seed_num': seed_num,
        'num_correct_trials': num_correct_trials,
        'num_perf_trials': num_perf_trials,
        'verbose': verbose,
        'measure_performance': measure_performance,
        'build_dir': build_dir,
        'device': device,
        'backend': backend,
        'dtype_str': dtype_str,
        'gpu_name': gpu_name,
        'level': level,
        'problem_id': problem_id,
    }
    args_json = json.dumps(args_dict)
    
    process = subprocess.Popen(
        [sys.executable, script_path, args_json],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=src_dir
    )
    
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return KernelExecResult(
            compiled=False,
            correctness=False,
            metadata={'error': 'Subprocess timeout', 'timeout': timeout}
        )
    
    stdout_str = stdout.decode('utf-8')
    stderr_str = stderr.decode('utf-8')
    
    if process.returncode != 0:
        try:
            error_dict = json.loads(stdout_str)
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={
                    'error': error_dict.get('error', 'Unknown error'),
                    'error_type': error_dict.get('error_type', 'Unknown'),
                    'traceback': error_dict.get('traceback', ''),
                    'stderr': stderr_str
                }
            )
        except json.JSONDecodeError:
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={
                    'error': 'Subprocess failed',
                    'stdout': stdout_str,
                    'stderr': stderr_str,
                    'returncode': process.returncode
                }
            )
    
    try:
        result_dict = json.loads(stdout_str)
        if result_dict is None:
            return None
        return KernelExecResult(**result_dict)
    except json.JSONDecodeError as e:
        return KernelExecResult(
            compiled=False,
            correctness=False,
            metadata={
                'error': 'Failed to parse result',
                'json_error': str(e),
                'stdout': stdout_str,
                'stderr': stderr_str
            }
        )


def wrapped_eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    verbose: bool = False,
    measure_performance: bool = False,
    build_dir: os.PathLike = None,
    device: Union[torch.device, int] = (
        torch.cuda.current_device() if torch.cuda.is_available() else None
    ),
    backend: str = "cuda",
    dtype_str: str = "fp32",
    use_remote_eval: bool = False,
    remote_eval_url: str = "http://127.0.0.1:12017",
    timeout: int = 1800,  # 30 minutes default timeout
    test_source: str = "KB",
    level: str = None,
    problem_id: str = None,
    gpu_name: str = None,
) -> KernelExecResult:
    """
    Wrapper for eval_kernel_against_ref that runs it in a subprocess to avoid RE (Runtime Errors)
    crashing the main process.
    
    All parameters are the same as eval_kernel_against_ref, with an additional timeout parameter.
    test_source: "KB", "SYN", "FIT" (flashinfer-trace), "MLSYS" (mlsys26-contest), or "TBG" (TritonBench-G)
    level: for FIT, this is op_type (e.g., "gemm")
    problem_id: for FIT, this is problem_name (e.g., "gemm_n128_k2048")
    """
    # Normalize device once for both remote and local paths.
    device_val = _normalize_device(device)

    # FIT/MLSYS evaluation requires remote evaluation
    if test_source in ("FIT", "MLSYS"):
        if not use_remote_eval:
            raise NotImplementedError(
                f"FIT/MLSYS evaluation requires remote evaluation server. "
                f"Set use_remote_eval=True and provide remote_eval_url. "
                f"Task: {level}/{problem_id}"
            )
        
        return _remote_eval_with_retry(
            url=f"{remote_eval_url}/evaluate_fit_kernel",
            json_data={
                "kernel_code": custom_model_src,
                "task_id": problem_id,
                "backend": backend,
                "timeout": timeout
            },
            timeout=timeout + 10,  # Add buffer for network overhead
            eval_type="FIT",
            raise_on_failure=True  # FIT raises on failure to match original behavior
        )

    # TritonBench-G evaluation requires remote evaluation.
    # Use the unified /evaluate endpoint and dispatch by test_source server-side.
    if test_source == "TBG":
        if not use_remote_eval:
            raise NotImplementedError(
                f"TBG evaluation requires remote evaluation server. "
                f"Set use_remote_eval=True and provide remote_eval_url. "
                f"Task: {problem_id}"
            )
        return _remote_eval_with_retry(
            url=f"{remote_eval_url}/evaluate",
            json_data={
                "test_source": "TBG",
                # Keep both pairs for compatibility with both generic and TBG-specific parsing.
                "original_model_src": original_model_src or "",
                "custom_model_src": custom_model_src,
                "kernel_code": custom_model_src,
                "task_id": str(problem_id),
                "problem_id": str(problem_id),
                "device": device_val,
                "backend": backend,
                "dtype_str": dtype_str,
                "timeout": timeout,
            },
            timeout=timeout + 10,
            eval_type="TBG",
            raise_on_failure=False,
        )
    
    # Remote evaluation
    if use_remote_eval:
        json_data = {
            "original_model_src": original_model_src,
            "custom_model_src": custom_model_src,
            "seed_num": seed_num,
            "num_correct_trials": num_correct_trials,
            "num_perf_trials": num_perf_trials,
            "verbose": verbose,
            "measure_performance": measure_performance,
            "build_dir": build_dir,
            "device": device_val,
            "backend": backend,
            "dtype_str": dtype_str,
            "gpu_name": gpu_name,
            "level": level,
            "problem_id": problem_id,
            "timeout": timeout,
        }
        return _remote_eval_with_retry(
            url=f"{remote_eval_url}/evaluate",
            json_data=json_data,
            timeout=timeout,
            eval_type="normal",
            raise_on_failure=True  # Normal eval returns error KernelExecResult
        )
    
    # Local subprocess evaluation
    return _local_subprocess_eval(
        original_model_src=original_model_src,
        custom_model_src=custom_model_src,
        seed_num=seed_num,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        verbose=verbose,
        measure_performance=measure_performance,
        build_dir=build_dir,
        device=device_val,
        backend=backend,
        dtype_str=dtype_str,
        timeout=timeout,
        gpu_name=gpu_name,
        level=level,
        problem_id=problem_id,
    )


def ncu_profile_solution(
    original_model_src: str,
    custom_model_src: str,
    *,
    device: Union[torch.device, int] = (
        torch.cuda.current_device() if torch.cuda.is_available() else None
    ),
    backend: str = "cuda",
    dtype_str: str = "fp32",
    seed_num: int = 42,
    build_dir: os.PathLike = None,
    # NCU configuration
    ncu_path: str = "ncu",
    set: str = None,
    sections: list[str] = ["SpeedOfLight"],
    kernel_name: str | None = None,
    page: str = "details",
    nvtx_range: str = "kernelbench_ncu_profile",
    # Execution control
    timeout: int = 120,
    tmpdir: str | None = None,
    max_lines: int | None = 4000,
) -> KernelExecResult:
    """
    Profile a KernelBench solution using NVIDIA Nsight Compute.

    This runs `ncu` against a standalone runner (`src.ncu_subprocess_runner`) that
    compiles + executes the custom model once inside an NVTX range, so NCU can filter.
    """
    device_val = str(device)
    if shutil.which(ncu_path) is None:
        return KernelExecResult(
            compiled=False,
            correctness=False,
            metadata={
                "error": f"NCU executable not found at '{ncu_path}'. Please install NVIDIA Nsight Compute."
            },
        )

    with tempfile.TemporaryDirectory(prefix="kb_ncu_", dir=tmpdir) as build_dir_tmp:
        payload_dir = Path(build_dir_tmp)
        # The NCU runner expects file paths in args.json; write sources into the payload dir.
        original_model_src_path = payload_dir / "original_model_src.py"
        custom_model_src_path = payload_dir / "custom_model_src.py"
        original_model_src_path.write_text(original_model_src)
        custom_model_src_path.write_text(custom_model_src)

        payload = {
            "original_model_src_path": str(original_model_src_path),
            "custom_model_src_path": str(custom_model_src_path),
            "seed_num": seed_num,
            "backend": backend,
            "dtype_str": dtype_str,
            "build_dir": str(build_dir) if build_dir is not None else None,
            "num_warmup": 1,
            "nvtx_range": nvtx_range,
        }
        (payload_dir / "args.json").write_text(json.dumps(payload))

        cmd: list[str] = [
            ncu_path,
            "--page",
            page,
            "--nvtx",
            "--nvtx-include",
            f"{nvtx_range}]",
            "--target-processes",
            "all",
        ]
        if set is not None:
            cmd.extend(["--set", set])
        if sections:
            for section in sections:
                cmd.extend(["--section", section])
        if kernel_name:
            cmd.extend(["--kernel-name", kernel_name])
        cmd.append("-f")
        cmd.extend(
            [
                sys.executable,
                "-u",
                "-m",
                "src.ncu_subprocess_runner",
                "--data-dir",
                str(payload_dir),
                "--device",
                f"cuda:{device_val}",
                "--backend",
                backend,
                "--dtype-str",
                dtype_str,
                "--seed",
                str(seed_num),
                "--nvtx-range",
                nvtx_range,
            ]
        )
        if build_dir is not None:
            cmd.extend(["--build-dir", str(build_dir)])

        env = os.environ.copy()
        if tmpdir:
            env["TMPDIR"] = tmpdir

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, env=env, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={
                    "error": f"NCU profiling timed out after {timeout} seconds.",
                    "ncu_cmd": " ".join(cmd),
                },
            )

        output = (result.stdout or "") + (result.stderr or "")
        if max_lines is not None:
            lines = output.splitlines()
            if len(lines) > max_lines:
                output = "\n".join(
                    lines[:max_lines]
                    + [f"[Output truncated: {len(lines) - max_lines} more lines]"]
                )

        if result.returncode != 0:
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={
                    "error": f"NCU exited with non-zero return code {result.returncode}.",
                    "ncu_cmd": " ".join(cmd),
                    "ncu_output": output,
                },
            )

        return KernelExecResult(
            compiled=True,
            correctness=True,
            metadata={
                "ncu_cmd": " ".join(cmd),
                "ncu_output": output,
            },
        )

if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="KernelBench eval / NCU test entrypoint")
    parser.add_argument(
        "--test-ncu",
        action="store_true",
        help="Run a minimal NCU profile test on the example solution.",
    )

# fetch_baseline_time("level1", 0, ["1_Square_matrix_multiplication_.py"], "tests/baseline_time_matx3.json")
    # Shared / example sources
    parser.add_argument(
        "--original-model-src-path",
        default="test_data/kernelbench/reference_src.py",
        help="Path to original model python source file.",
    )
    parser.add_argument(
        "--custom-model-src-path",
        default="test_data/kernelbench/global_best_kernel_50.py",
        help="Path to custom model python source file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="triton", help="cuda|triton")
    parser.add_argument("--dtype-str", default="fp32", help="fp16|fp32|bf16")

    # Remote-eval demo args (kept as the default behavior)
    parser.add_argument("--remote-eval-url", default="http://localhost:12017")

    # NCU args (only used when --test-ncu is set)
    parser.add_argument("--device", default="cuda:0", help="CUDA device, e.g. cuda:0")
    parser.add_argument("--ncu-path", default="ncu", help="Path to Nsight Compute executable")
    parser.add_argument("--ncu-set", default="basic", help="NCU --set preset")
    parser.add_argument(
        "--ncu-section",
        action="append",
        default=["SpeedOfLight"],
        help="Repeatable: NCU --section <name> (e.g. SpeedOfLight).",
    )
    parser.add_argument("--ncu-page", default="details", help="NCU --page (e.g. details)")
    parser.add_argument("--kernel-name", default=None, help="Optional NCU --kernel-name filter")
    parser.add_argument("--nvtx-range", default="kernelbench_ncu_profile")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--tmpdir", default=None, help="Optional TMPDIR for compilation/profiling")
    parser.add_argument("--max-lines", type=int, default=4000, help="Max NCU output lines to keep")
    args = parser.parse_args()

    original_model_src = open(args.original_model_src_path, "r").read()
    custom_model_src = open(args.custom_model_src_path, "r").read()

    if args.test_ncu:
        result = ncu_profile_solution(
            original_model_src=original_model_src,
            custom_model_src=custom_model_src,
            device=args.device,
            backend=args.backend,
            dtype_str=args.dtype_str,
            seed_num=args.seed,
            ncu_path=args.ncu_path,
            set=args.ncu_set,
            sections=args.ncu_section,
            kernel_name=args.kernel_name,
            page=args.ncu_page,
            nvtx_range=args.nvtx_range,
            timeout=args.timeout,
            tmpdir=args.tmpdir,
            max_lines=args.max_lines,
        )
        print(result.metadata["ncu_output"])
    else:
        # Backwards-compatible demo behavior: remote eval.
        result = wrapped_eval_kernel_against_ref(
            original_model_src=original_model_src,
            custom_model_src=custom_model_src,
            seed_num=args.seed,
            num_correct_trials=1,
            num_perf_trials=10,
            verbose=False,
            backend=args.backend,
            measure_performance=True,
            use_remote_eval=True,
            remote_eval_url=args.remote_eval_url,
        )
        print(result)
        print(type(result))
