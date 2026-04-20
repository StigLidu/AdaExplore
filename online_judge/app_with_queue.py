"""
Online Judge API Server with Queue and Concurrency Control
Handles high-concurrency scenarios with request queuing and resource management
"""

import os
import sys
import time
import traceback
import asyncio
import subprocess
import json
import logging
import ast
import tempfile
import base64
from contextlib import asynccontextmanager
from typing import Optional, Dict, List
from collections import defaultdict
import torch
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging based on DEBUG mode
DEBUG_MODE = os.environ.get("DEBUG", "false").lower() == "true"
LOG_LEVEL = logging.DEBUG if DEBUG_MODE else logging.WARNING

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

if DEBUG_MODE:
    logger.debug("Debug mode enabled - showing all log levels")
else:
    logger.warning("Normal mode - showing WARNING and ERROR logs only")

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.format import KernelExecResult, REPO_TOP_PATH

# FlashInfer-style datasets (FIT and FIT-compatible datasets like MLSYS'26 contest)
FLASHINFER_TRACE_DATASET_ROOT = os.path.join(REPO_TOP_PATH, "datasets", "flashinfer-trace")
MLSYS26_CONTEST_DATASET_ROOT = os.path.join(REPO_TOP_PATH, "datasets", "mlsys26-contest")
TRITONBENCH_G_DATASET_ROOT = os.path.join(REPO_TOP_PATH, "datasets", "TritonBench", "data", "TritonBench_G_v1")
TRITONBENCH_G_FALLBACK_ROOT = os.path.join(REPO_TOP_PATH, "datasets", "TritonBench_G_v1")
TRITONBENCH_T_DATASET_ROOT = os.path.join(REPO_TOP_PATH, "datasets", "TritonBench", "data", "TritonBench_T_v1")
TRITONBENCH_T_FALLBACK_ROOT = os.path.join(REPO_TOP_PATH, "datasets", "TritonBench_T_v1")

FLASHINFER_STYLE_DATASET_ROOTS: List[str] = [FLASHINFER_TRACE_DATASET_ROOT, MLSYS26_CONTEST_DATASET_ROOT]

def _list_flashinfer_op_types(dataset_root: str) -> list[str]:
    defs_dir = os.path.join(dataset_root, "definitions")
    if not os.path.isdir(defs_dir):
        return []
    return sorted(
        d for d in os.listdir(defs_dir)
        if os.path.isdir(os.path.join(defs_dir, d))
    )

FLASHINFER_TRACE_OP_TYPES = ["gemm", "gqa_paged", "gqa_ragged", "mla_paged", "moe", "rmsnorm", "sampling", "dsa_paged", "gdn"]

def _get_tritonbench_dataset_root(benchmark: str) -> str:
    benchmark = str(benchmark).upper()
    if benchmark == "TBG":
        env_var = "TRITONBENCH_G_PATH"
        default_root = TRITONBENCH_G_DATASET_ROOT
        fallback_root = TRITONBENCH_G_FALLBACK_ROOT
    elif benchmark == "TBT":
        env_var = "TRITONBENCH_T_PATH"
        default_root = TRITONBENCH_T_DATASET_ROOT
        fallback_root = TRITONBENCH_T_FALLBACK_ROOT
    else:
        raise ValueError(f"Unsupported TritonBench benchmark: {benchmark}")

    env_root = os.environ.get(env_var)
    candidates = []
    if env_root:
        candidates.append(env_root)
    candidates.extend([default_root, fallback_root])
    for root in candidates:
        if os.path.isdir(root):
            return root
    return default_root

def _resolve_tritonbench_task_path(task_id: str, benchmark: str) -> str:
    dataset_root = _get_tritonbench_dataset_root(benchmark)
    if not os.path.isdir(dataset_root):
        expected_subdir = "TritonBench_G_v1" if benchmark == "TBG" else "TritonBench_T_v1"
        raise FileNotFoundError(
            f"{benchmark} dataset root does not exist: {dataset_root}. "
            f"Set TRITONBENCH_{benchmark[-1]}_PATH or place data under "
            f"datasets/TritonBench/data/{expected_subdir}."
        )

    py_files = sorted([f for f in os.listdir(dataset_root) if f.endswith(".py")])
    if not py_files:
        raise FileNotFoundError(f"No .py tasks found in {benchmark} dataset root: {dataset_root}")

    task = str(task_id).strip()
    if task.isdigit():
        idx = int(task) - 1
        if idx < 0 or idx >= len(py_files):
            raise ValueError(f"{benchmark} task index out of range: {task_id}, valid 1..{len(py_files)}")
        return os.path.join(dataset_root, py_files[idx])

    filename = task if task.endswith(".py") else f"{task}.py"
    if filename not in py_files:
        raise ValueError(f"{benchmark} task not found: {task_id}")
    return os.path.join(dataset_root, filename)


def _resolve_tbg_task_path(task_id: str) -> str:
    return _resolve_tritonbench_task_path(task_id, "TBG")


def _resolve_tbt_task_path(task_id: str) -> str:
    return _resolve_tritonbench_task_path(task_id, "TBT")

def _split_tbg_reference_source(reference_src: str) -> tuple[str, str]:
    separator = "#" * 146
    if separator in reference_src:
        reference_prelude, test_block = reference_src.split(separator, 1)
    else:
        reference_prelude = ""
        test_block = reference_src
    if "def test_" not in test_block:
        raise ValueError("TritonBench task file does not contain built-in test block (`def test_...`).")
    return reference_prelude, test_block

def _extract_tbg_reference_imports(reference_prelude: str) -> str:
    source = str(reference_prelude or "")
    if not source.strip():
        return ""

    try:
        module = ast.parse(source)
        imports = []
        for node in module.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                segment = ast.get_source_segment(source, node)
                if segment:
                    imports.append(segment.strip())
        if imports:
            return "\n".join(imports)
    except Exception:
        pass

    # Fallback for malformed source.
    imports = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(line.rstrip())
    return "\n".join(imports)

def _sanitize_tbg_candidate_code(code: str) -> str:
    text = str(code or "")
    if "```python" in text:
        text = text.split("```python", 1)[1]
    if "```" in text:
        text = text.split("```", 1)[0]
    text = text.replace("<|im_end|>", "").replace("<|EOT|>", "").strip()

    try:
        ast.parse(text)
        return text
    except Exception:
        pass

    # Fallback: drop leading explanation lines and keep probable code section.
    lines = text.splitlines()
    starters = ("import ", "from ", "def ", "class ", "@", "if ", "for ", "while ", "try:")
    start_idx = 0
    while start_idx < len(lines):
        if lines[start_idx].lstrip().startswith(starters):
            break
        start_idx += 1
    candidate = "\n".join(lines[start_idx:]).strip() if start_idx < len(lines) else text
    try:
        ast.parse(candidate)
        return candidate
    except Exception:
        return text


def _prepare_tbg_test_block(test_block: str) -> tuple[str, str]:
    """
    Return a sanitized test block that defines the official test function but
    strips top-level auto-execution like `result_gold = test_xxx()`.
    """
    source = str(test_block or "")
    module = ast.parse(source)

    test_names = [
        node.name
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    if not test_names:
        raise ValueError("TBG task file does not define any `test_...` function.")

    def _called_name(node):
        value = None
        if isinstance(node, ast.Expr):
            value = node.value
        elif isinstance(node, ast.Assign):
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            return value.func.id
        return None

    filtered_body = []
    for node in module.body:
        called_name = _called_name(node)
        if called_name in test_names:
            continue
        filtered_body.append(node)

    module.body = filtered_body
    ast.fix_missing_locations(module)
    return ast.unparse(module).strip() + "\n", test_names[0]

def _run_tritonbench_evaluation_sync(
    kernel_code: str,
    task_id: str,
    device: int,
    timeout: int,
    measure_performance: bool,
    num_perf_trials: int,
    *,
    benchmark: str,
    resolve_task_path,
) -> KernelExecResult:
    """
    Evaluate a TritonBench candidate by prepending reference imports and appending
    the official built-in test block. Optionally measure end-to-end speedup by
    timing the official test function for both the candidate and the reference
    implementation on the same seeds.
    """
    task_path = resolve_task_path(task_id)
    with open(task_path, "r", encoding="utf-8") as f:
        reference_src = f.read()

    reference_prelude, test_block = _split_tbg_reference_source(reference_src)
    reference_imports = _extract_tbg_reference_imports(reference_prelude)
    candidate_code = _sanitize_tbg_candidate_code(kernel_code)
    prepared_test_block, test_fn_name = _prepare_tbg_test_block(test_block)

    candidate_source = "\n\n".join(
        section.strip("\n")
        for section in (reference_imports, candidate_code, prepared_test_block)
        if section and section.strip()
    ) + "\n"
    reference_source = "\n\n".join(
        section.strip("\n")
        for section in (reference_imports, reference_prelude, prepared_test_block)
        if section and section.strip()
    ) + "\n"

    candidate_source_b64 = base64.b64encode(candidate_source.encode("utf-8")).decode("ascii")
    reference_source_b64 = base64.b64encode(reference_source.encode("utf-8")).decode("ascii")
    perf_trials = max(1, int(num_perf_trials or 1))

    script = f"""
import base64
import json
import random
import time

import numpy as np
import torch

CANDIDATE_SOURCE_B64 = {candidate_source_b64!r}
REFERENCE_SOURCE_B64 = {reference_source_b64!r}
TEST_FN_NAME = {test_fn_name!r}
MEASURE_PERFORMANCE = {bool(measure_performance)!r}
NUM_PERF_TRIALS = {perf_trials}

def _decode_source(encoded):
    return base64.b64decode(encoded.encode("ascii")).decode("utf-8")

def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _load_namespace(source, module_name):
    ns = {{"__name__": module_name}}
    exec(compile(source, module_name, "exec"), ns, ns)
    return ns

def _benchmark(fn, trials, seed_base):
    warmup = min(3, max(1, trials))
    for i in range(warmup):
        _set_seed(seed_base + i)
        fn()
        _sync()

    times_ms = []
    for i in range(trials):
        _set_seed(seed_base + 1000 + i)
        _sync()
        start = time.perf_counter()
        fn()
        _sync()
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return times_ms

candidate_ns = _load_namespace(_decode_source(CANDIDATE_SOURCE_B64), "__tbg_candidate__")
candidate_fn = candidate_ns[TEST_FN_NAME]

_set_seed(0)
candidate_fn()
_sync()

payload = {{
    "compiled": True,
    "correctness": True,
    "runtime_ms": -1.0,
    "runtime_stats": {{"fast_p": 0.0}},
}}

if MEASURE_PERFORMANCE:
    reference_ns = _load_namespace(_decode_source(REFERENCE_SOURCE_B64), "__tbg_reference__")
    reference_fn = reference_ns[TEST_FN_NAME]
    ref_times = _benchmark(reference_fn, NUM_PERF_TRIALS, 1234)
    cand_times = _benchmark(candidate_fn, NUM_PERF_TRIALS, 1234)
    ref_mean = sum(ref_times) / len(ref_times)
    cand_mean = sum(cand_times) / len(cand_times)
    speedup = (ref_mean / cand_mean) if cand_mean > 0 else 0.0
    payload["runtime_ms"] = cand_mean
    payload["runtime_stats"] = {{
        "reference_latency_ms": ref_mean,
        "candidate_latency_ms": cand_mean,
        "reference_latency_min_ms": min(ref_times),
        "candidate_latency_min_ms": min(cand_times),
        "reference_latency_max_ms": max(ref_times),
        "candidate_latency_max_ms": max(cand_times),
        "num_perf_trials": NUM_PERF_TRIALS,
        "fast_p": speedup,
        "speedup_factor": speedup,
    }}

print(json.dumps(payload))
""".strip() + "\n"

    with tempfile.TemporaryDirectory(prefix="tbg_eval_") as tmp_dir:
        temp_script = os.path.join(tmp_dir, os.path.basename(task_path))
        with open(temp_script, "w", encoding="utf-8") as f:
            f.write(script)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        env["PYTHONUNBUFFERED"] = "1"

        try:
            result = subprocess.run(
                [sys.executable, temp_script],
                capture_output=True,
                text=True,
                cwd=REPO_TOP_PATH,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return KernelExecResult(
                compiled=False,
                correctness=False,
                runtime=-1.0,
                runtime_stats={"fast_p": 0.0},
                metadata={
                    "task_id": str(task_id),
                    "benchmark": benchmark,
                    "task_path": task_path,
                    "error": f"{benchmark} evaluation timeout: exceeded {timeout}s",
                },
            )

    stderr = result.stderr or ""
    stdout = result.stdout or ""
    compile_error_tokens = ("SyntaxError", "IndentationError", "TabError")
    compiled = result.returncode == 0 or not any(tok in stderr for tok in compile_error_tokens)
    correctness = result.returncode == 0
    runtime = -1.0
    runtime_stats = {"fast_p": 0.0}

    if correctness:
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                runtime = float(payload.get("runtime_ms", -1.0))
                runtime_stats = payload.get("runtime_stats", {}) or {"fast_p": 0.0}
                break
            except Exception:
                continue

    return KernelExecResult(
        compiled=compiled,
        correctness=correctness,
        runtime=runtime,
        runtime_stats=runtime_stats,
        metadata={
            "task_id": str(task_id),
            "benchmark": benchmark,
            "task_path": task_path,
            "returncode": result.returncode,
            "stdout": stdout[-4000:],
            "stderr": stderr[-4000:],
            "measure_performance": bool(measure_performance),
            "num_perf_trials": perf_trials,
        },
    )


def run_tbg_evaluation_sync(
    kernel_code: str,
    task_id: str,
    device: int,
    timeout: int = 300,
    measure_performance: bool = False,
    num_perf_trials: int = 10,
) -> KernelExecResult:
    return _run_tritonbench_evaluation_sync(
        kernel_code,
        task_id,
        device,
        timeout,
        measure_performance,
        num_perf_trials,
        benchmark="TBG",
        resolve_task_path=_resolve_tbg_task_path,
    )


def run_tbt_evaluation_sync(
    kernel_code: str,
    task_id: str,
    device: int,
    timeout: int = 300,
    measure_performance: bool = False,
    num_perf_trials: int = 10,
) -> KernelExecResult:
    return _run_tritonbench_evaluation_sync(
        kernel_code,
        task_id,
        device,
        timeout,
        measure_performance,
        num_perf_trials,
        benchmark="TBT",
        resolve_task_path=_resolve_tbt_task_path,
    )


# Configuration
MAX_CONCURRENT_EVALS = int(os.environ.get("MAX_CONCURRENT_EVALS", "1"))  # Max concurrent evaluations per GPU
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", "100"))  # Max queued requests
RUNTIME_TIMEOUT = int(os.environ.get("RUNTIME_TIMEOUT", "600"))  # Max evaluation runtime after acquiring device (seconds)

# Parse available GPU IDs from environment variable (comma-separated, e.g., "0,1" or "0,1,2")
AVAILABLE_GPUS_ENV = os.environ.get("AVAILABLE_GPUS", None)
if AVAILABLE_GPUS_ENV:
    AVAILABLE_GPUS = [int(gpu_id.strip()) for gpu_id in AVAILABLE_GPUS_ENV.split(",") if gpu_id.strip().isdigit()]
else:
    AVAILABLE_GPUS = None  # None means all available GPUs

# GPU allocation mode: "auto" (automatically assign to least busy GPU) or "manual" (use specified device)
# In "auto" mode, the device parameter from request is ignored
GPU_ALLOCATION_MODE = os.environ.get("GPU_ALLOCATION_MODE", "auto").lower()
if GPU_ALLOCATION_MODE not in ("auto", "manual"):
    logger.warning(f"Invalid GPU_ALLOCATION_MODE '{GPU_ALLOCATION_MODE}', defaulting to 'auto'")
    GPU_ALLOCATION_MODE = "auto"


# Request Models
class EvaluationRequest(BaseModel):
    """Request model for kernel evaluation"""
    original_model_src: str = Field(default="", description="Reference architecture source code")
    custom_model_src: str = Field(default="", description="Custom kernel source code to evaluate")
    seed_num: int = Field(default=42, description="Random seed for reproducibility")
    num_correct_trials: int = Field(default=1, ge=1, description="Number of correctness trials")
    num_perf_trials: int = Field(default=10, ge=1, description="Number of performance trials")
    measure_performance: bool = Field(default=False, description="Whether to measure performance")
    backend: str = Field(default="cuda", pattern="^(cuda|triton)$", description="Backend: 'cuda' or 'triton'")
    dtype_str: str = Field(default="fp32", pattern="^(fp32|fp16|bf16)$", description="Data type: 'fp32', 'fp16', or 'bf16'")
    device: Optional[int] = Field(default=None, description="CUDA device ID (None for current device)")
    verbose: bool = Field(default=False, description="Verbose output")
    timeout: Optional[int] = Field(default=300, ge=1, le=3600, description="Timeout in seconds (max 3600)")
    build_dir: Optional[str] = Field(default=None, description="Build directory for compilation artifacts")
    gpu_name: Optional[str] = Field(default=None, description="GPU name")
    level: Optional[str] = Field(default=None, description="Level")
    problem_id: Optional[str] = Field(default=None, description="Problem ID")
    test_source: Optional[str] = Field(default=None, description="Benchmark source (e.g., KB/FIT/TBG/TBT)")
    kernel_code: Optional[str] = Field(default=None, description="Alias of custom_model_src for benchmark-specific eval")
    task_id: Optional[str] = Field(default=None, description="Task ID for benchmark-specific eval")


class FITEvaluationRequest(BaseModel):
    """Request model for FlashInfer Trace (FIT) evaluation"""
    kernel_code: str = Field(..., description="Kernel source code to evaluate")
    task_id: str = Field(..., description="Task ID (e.g., 'gemm_n128_k2048')")
    backend: str = Field(default="triton", pattern="^(cuda|triton)$", description="Backend: 'cuda' or 'triton'")
    timeout: Optional[int] = Field(default=300, ge=1, le=3600, description="Timeout in seconds (max 3600)")
    device: Optional[int] = Field(default=None, description="CUDA device ID (None for auto allocation)")

class TBGEvaluationRequest(BaseModel):
    """Request model for TritonBench-G evaluation"""
    kernel_code: str = Field(..., description="Kernel source code to evaluate")
    task_id: str = Field(..., description="Task ID (e.g., 'add_example' or 'add_example.py')")
    backend: str = Field(default="triton", pattern="^(cuda|triton)$", description="Backend: 'cuda' or 'triton'")
    num_perf_trials: int = Field(default=10, ge=1, description="Number of performance trials")
    measure_performance: bool = Field(default=False, description="Whether to measure performance")
    timeout: Optional[int] = Field(default=300, ge=1, le=3600, description="Timeout in seconds (max 3600)")
    device: Optional[int] = Field(default=None, description="CUDA device ID (None for auto allocation)")


class TBTEvaluationRequest(BaseModel):
    """Request model for TritonBench-T evaluation"""
    kernel_code: str = Field(..., description="Kernel source code to evaluate")
    task_id: str = Field(..., description="Task ID (e.g., 'add' or 'add.py')")
    backend: str = Field(default="triton", pattern="^(cuda|triton)$", description="Backend: 'cuda' or 'triton'")
    num_perf_trials: int = Field(default=10, ge=1, description="Number of performance trials")
    measure_performance: bool = Field(default=False, description="Whether to measure performance")
    timeout: Optional[int] = Field(default=300, ge=1, le=3600, description="Timeout in seconds (max 3600)")
    device: Optional[int] = Field(default=None, description="CUDA device ID (None for auto allocation)")

class EvaluationResponse(BaseModel):
    """Response model for kernel evaluation"""
    success: bool
    compiled: bool
    correctness: bool
    runtime: float = Field(default=-1.0, description="Runtime in microseconds")
    runtime_stats: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    execution_time: float = Field(description="Total API execution time in seconds")
    queue_time: float = Field(default=0.0, description="Time spent in queue (seconds)")
    error: Optional[str] = Field(default=None, description="Error message if evaluation failed")


class QueueStatusResponse(BaseModel):
    """Response model for queue status"""
    total_queued: int
    total_processing: int
    per_device: Dict[int, Dict[str, int]]
    max_concurrent: int
    max_queue_size: int


# Global state for queue management
class ResourceManager:
    """Manages GPU resources and request queues"""
    
    def __init__(self, max_concurrent_per_device: int = MAX_CONCURRENT_EVALS, available_gpus: Optional[List[int]] = None):
        self.max_concurrent = max_concurrent_per_device
        self.device_semaphores: Dict[int, asyncio.Semaphore] = {}
        self.processing_count: Dict[int, int] = defaultdict(int)
        self.waiting_count: Dict[int, int] = defaultdict(int)  # Count of requests waiting for semaphore
        self.lock = asyncio.Lock()
        self.available_gpus = available_gpus  # List of allowed GPU IDs, None means all
        
        # Initialize semaphores for specified GPUs only
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            gpu_list = available_gpus if available_gpus is not None else list(range(num_devices))
            
            # Validate GPU IDs
            for gpu_id in gpu_list:
                if gpu_id < 0 or gpu_id >= num_devices:
                    raise ValueError(f"Invalid GPU ID: {gpu_id}. Available GPUs: 0-{num_devices-1}")
            
            # Initialize semaphores only for allowed GPUs
            for device_id in gpu_list:
                self.device_semaphores[device_id] = asyncio.Semaphore(max_concurrent_per_device)
    
    def is_device_available(self, device_id: int) -> bool:
        """Check if a device ID is available for use"""
        if self.available_gpus is None:
            return device_id in self.device_semaphores
        return device_id in self.available_gpus and device_id in self.device_semaphores
    
    async def acquire_device(self, device_id: int) -> bool:
        """Acquire a device slot, returns True if acquired immediately, False if queued"""
        async with self.lock:
            if self.processing_count[device_id] < self.max_concurrent:
                self.processing_count[device_id] += 1
                logger.debug(f"Acquired device {device_id}, processing count: {self.processing_count[device_id]}, max concurrent: {self.max_concurrent}")
                return True
            return False
    
    async def wait_for_device(self, device_id: int, timeout: float = RUNTIME_TIMEOUT):
        """Wait for device availability with timeout - uses semaphore for concurrency control"""
        start_time = time.time()
        
        # Use semaphore to control concurrency
        # The semaphore ensures only MAX_CONCURRENT requests run at once
        # asyncio.Semaphore provides FIFO-like behavior for waiting coroutines
        semaphore = self.device_semaphores[device_id]
        
        # Check queue size limit
        async with self.lock:
            if self.waiting_count[device_id] >= MAX_QUEUE_SIZE:
                logger.warning(f"Queue full on device {device_id}: {self.waiting_count[device_id]}/{MAX_QUEUE_SIZE}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Queue is full. Max queue size: {MAX_QUEUE_SIZE}, currently waiting: {self.waiting_count[device_id]}"
                )
        
        # Try to acquire semaphore immediately with very short timeout
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=0.001)
            # Successfully acquired semaphore immediately
            async with self.lock:
                self.processing_count[device_id] += 1
                logger.debug(f"Acquired device {device_id} immediately, processing count: {self.processing_count[device_id]}, max concurrent: {self.max_concurrent}")
                return 0.0
        except asyncio.TimeoutError:
            # Semaphore not immediately available, need to wait
            async with self.lock:
                self.waiting_count[device_id] += 1
                logger.debug(f"Device {device_id} busy, queued. Processing: {self.processing_count[device_id]}, Waiting: {self.waiting_count[device_id]}, Max concurrent: {self.max_concurrent}")
        
        try:
            # Wait for semaphore (blocks until slot available)
            # asyncio.Semaphore wakes up waiting coroutines in FIFO order
            await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
            
            # Got the semaphore, now mark as processing
            async with self.lock:
                self.processing_count[device_id] += 1
                self.waiting_count[device_id] = max(0, self.waiting_count[device_id] - 1)
                logger.debug(f"Acquired device {device_id} after wait, processing count: {self.processing_count[device_id]}, waiting count: {self.waiting_count[device_id]}, max concurrent: {self.max_concurrent}")
            
            queue_time = time.time() - start_time
            return queue_time
            
        except asyncio.TimeoutError:
            async with self.lock:
                self.waiting_count[device_id] = max(0, self.waiting_count[device_id] - 1)
            logger.warning(f"Request timeout on device {device_id}: no slot available within {timeout}s")
            raise HTTPException(
                status_code=408,
                detail=f"Request timeout: No available GPU slot within {timeout}s"
            )
    
    async def release_device(self, device_id: int):
        """Release a device slot"""
        try:
            async with self.lock:
                self.processing_count[device_id] = max(0, self.processing_count[device_id] - 1)
                logger.debug(f"Released device {device_id}, processing count: {self.processing_count[device_id]}, waiting count: {self.waiting_count[device_id]}, max concurrent: {self.max_concurrent}")
            
            # Release semaphore to allow next queued request
            self.device_semaphores[device_id].release()
        except Exception as e:
            logger.error(f"Failed to release device {device_id}: {e}")
            raise
    
    def get_least_busy_device(self) -> int:
        """
        Get the device with the lowest load (processing + waiting).
        Used for automatic GPU allocation mode.
        """
        if not self.device_semaphores:
            raise HTTPException(
                status_code=503,
                detail="No GPU devices available"
            )
        
        min_load = float('inf')
        best_device = None
        
        for device_id in self.device_semaphores.keys():
            # Load = currently processing + waiting in queue
            load = self.processing_count[device_id] + self.waiting_count[device_id]
            if load < min_load:
                min_load = load
                best_device = device_id
        
        return best_device
    
    def get_status(self) -> QueueStatusResponse:
        """Get current queue and processing status"""
        total_queued = sum(self.waiting_count.values())
        total_processing = sum(self.processing_count.values())
        
        per_device = {}
        if torch.cuda.is_available():
            device_list = self.available_gpus if self.available_gpus is not None else list(range(torch.cuda.device_count()))
            for device_id in device_list:
                per_device[device_id] = {
                    "queued": self.waiting_count[device_id],
                    "processing": self.processing_count[device_id],
                    "max_concurrent": self.max_concurrent
                }
        
        return QueueStatusResponse(
            total_queued=total_queued,
            total_processing=total_processing,
            per_device=per_device,
            max_concurrent=self.max_concurrent,
            max_queue_size=MAX_QUEUE_SIZE
        )


# Global resource manager
resource_manager = ResourceManager(available_gpus=AVAILABLE_GPUS)


# Global state
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. The judge will not function properly.")
    else:
        if DEBUG_MODE:
            logger.debug(f"CUDA available. Total devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                status = "✓ Available" if resource_manager.is_device_available(i) else "✗ Not available"
                logger.debug(f"  Device {i}: {torch.cuda.get_device_name(i)} [{status}]")
            if AVAILABLE_GPUS is not None:
                logger.debug(f"Available GPUs (restricted): {AVAILABLE_GPUS}")
            else:
                logger.debug(f"Available GPUs: All ({list(range(torch.cuda.device_count()))})")
            logger.debug(f"Max concurrent evaluations per device: {MAX_CONCURRENT_EVALS}")
            logger.debug(f"Max queue size per device: {MAX_QUEUE_SIZE}")
            logger.debug(f"GPU allocation mode: {GPU_ALLOCATION_MODE} ({'auto-assign to least busy GPU' if GPU_ALLOCATION_MODE == 'auto' else 'use specified device'})")
    
    # Start background task to log queue status every 2 minutes
    async def log_queue_status():
        while True:
            try:
                await asyncio.sleep(120)  # 2 minutes
                status = resource_manager.get_status()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"[{timestamp}] Queue Status - Queued: {status.total_queued}, Processing: {status.total_processing}")
                for device_id, info in status.per_device.items():
                    logger.info(f"  Device {device_id}: Queued={info['queued']}, Processing={info['processing']}/{info['max_concurrent']}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue status logging task: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    queue_task = asyncio.create_task(log_queue_status())
    
    yield
    
    # Shutdown
    queue_task.cancel()
    try:
        await queue_task
    except asyncio.CancelledError:
        pass
    
    torch.cuda.empty_cache()
    if DEBUG_MODE:
        logger.debug("Shutting down judge server...")


# Create FastAPI app
app = FastAPI(
    title="KernelBench Online Judge (With Queue)",
    description="REST API for evaluating CUDA/Triton kernels with queue management and concurrency control",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with detailed logging"""
    errors = exc.errors()
    error_details = []
    for error in errors:
        error_details.append({
            "field": ".".join(str(loc) for loc in error.get("loc", [])),
            "message": error.get("msg", ""),
            "type": error.get("type", ""),
            "input": error.get("input", "")
        })
    
    logger.error(f"Validation error for {request.url.path}: {error_details}")
    logger.error(f"Request body: {await request.body()}")
    
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Request validation failed",
            "errors": error_details
        }
    )


@app.get("/")
async def root():
    """Health check endpoint"""
    status = resource_manager.get_status()
    available_gpus = resource_manager.available_gpus if resource_manager.available_gpus else list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    return {
        "status": "online",
        "service": "KernelBench Online Judge (With Queue)",
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "available_gpus": available_gpus,
        "gpu_allocation_mode": GPU_ALLOCATION_MODE,
        "queue_status": status.dict()
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    health_info = {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "gpu_allocation_mode": GPU_ALLOCATION_MODE,
    }
    
    if torch.cuda.is_available():
        health_info["device_count"] = torch.cuda.device_count()
        available_gpus = resource_manager.available_gpus if resource_manager.available_gpus else list(range(torch.cuda.device_count()))
        health_info["available_gpus"] = available_gpus
        health_info["devices"] = [
            {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "available": resource_manager.is_device_available(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(i),
                "memory_reserved": torch.cuda.memory_reserved(i),
            }
            for i in range(torch.cuda.device_count())
        ]
    
    # Add queue status
    status = resource_manager.get_status()
    health_info["queue_status"] = status.dict()
    
    return health_info


@app.get("/queue/status", response_model=QueueStatusResponse)
async def queue_status():
    """Get current queue and processing status"""
    return resource_manager.get_status()


def run_evaluation_sync(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int,
    num_correct_trials: int,
    num_perf_trials: int,
    verbose: bool,
    measure_performance: bool,
    build_dir: Optional[str],
    device: int,
    backend: str,
    dtype_str: str,
    timeout: int = 300,
    gpu_name: str = None,
    level: str = None,
    problem_id: str = None,
) -> KernelExecResult:
    """
    Synchronous wrapper for eval_kernel_against_ref using subprocess
    This runs in a subprocess to avoid runtime errors crashing the main process
    Similar to wrapped_eval_kernel_against_ref
    """
    # Get the path to the subprocess runner script
    # Find the src directory
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to parent directory (repo root)
    repo_root = os.path.dirname(current_file_dir)
    eval_file_dir = os.path.join(repo_root, 'src')
    
    # Use the separate script file
    script_path = os.path.join(eval_file_dir, 'eval_subprocess_runner.py')
    if not os.path.exists(script_path):
        raise FileNotFoundError(
            f"Subprocess runner script not found at {script_path}"
        )
    
    # Prepare arguments as JSON
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
    
    # Run the subprocess
    process = subprocess.Popen(
        [sys.executable, script_path, args_json],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=repo_root
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
    
    # Parse the result
    stdout_str = stdout.decode('utf-8')
    stderr_str = stderr.decode('utf-8')
    
    if process.returncode != 0:
        # Try to parse error from stdout
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
    
    # Parse successful result
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


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_kernel(request: EvaluationRequest):
    """
    Evaluate a custom kernel against a reference implementation
    
    This endpoint:
    1. Queues the request if all GPU slots are busy
    2. Compiles the custom kernel
    3. Checks correctness against the reference
    4. Optionally measures performance
    
    Returns evaluation results including compilation status, correctness, and performance metrics.
    """
    queue_time = 0.0
    total_start_time = time.time()
    queue_start_time = time.time()
    test_source = str(request.test_source or "").strip().upper()
    is_tbg_eval = test_source in {"TBG", "TRITONBENCH_G", "TRITONBENCH-G"}
    is_tbt_eval = test_source in {"TBT", "TRITONBENCH_T", "TRITONBENCH-T"}
    device_acquired = False
    
    if DEBUG_MODE:
        logger.debug(
            f"Received evaluation request: test_source={test_source or 'KB'}, device={request.device}, "
            f"backend={request.backend}, dtype={request.dtype_str}, "
            f"num_correct_trials={request.num_correct_trials}, num_perf_trials={request.num_perf_trials}"
        )
    
    try:
        # Validate CUDA availability
        if not torch.cuda.is_available():
            raise HTTPException(
                status_code=503,
                detail="CUDA is not available. Cannot evaluate kernels."
            )
        
        # Determine device based on allocation mode
        if GPU_ALLOCATION_MODE == "auto":
            # Auto mode: automatically assign to least busy GPU, ignore request.device
            device = resource_manager.get_least_busy_device()
            if DEBUG_MODE:
                logger.debug(f"Auto allocation mode: assigned to device {device} (least busy)")
        else:
            # Manual mode: use specified device or default
            device = request.device
            if device is None:
                device = torch.cuda.current_device()
            
            # Validate device
            if device >= torch.cuda.device_count():
                error_msg = f"Device {device} not available. Only {torch.cuda.device_count()} device(s) available."
                logger.warning(f"400 Bad Request: {error_msg}")
                raise HTTPException(
                    status_code=400,
                    detail=error_msg
                )
            
            # Check if device is in the allowed list
            if not resource_manager.is_device_available(device):
                available_list = resource_manager.available_gpus if resource_manager.available_gpus else list(range(torch.cuda.device_count()))
                error_msg = f"Device {device} is not available. Available GPUs: {available_list}"
                logger.warning(f"400 Bad Request: {error_msg}")
                raise HTTPException(
                    status_code=400,
                    detail=error_msg
                )
        
        # queue_timeout: how long to wait for a device (use request.timeout directly, can be long)
        # runtime_timeout: max evaluation runtime after acquiring device (capped at RUNTIME_TIMEOUT)
        queue_timeout = request.timeout or RUNTIME_TIMEOUT
        runtime_timeout = min(request.timeout or RUNTIME_TIMEOUT, RUNTIME_TIMEOUT)
        queue_time = await resource_manager.wait_for_device(device, timeout=queue_timeout)
        device_acquired = True
        queue_end_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()

            if is_tbg_eval or is_tbt_eval:
                tb_kernel_code = request.kernel_code or request.custom_model_src
                tb_task_id = request.task_id or request.problem_id
                benchmark_name = "TBG" if is_tbg_eval else "TBT"
                eval_fn = run_tbg_evaluation_sync if is_tbg_eval else run_tbt_evaluation_sync
                if not tb_kernel_code:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{benchmark_name} evaluation via /evaluate requires kernel_code or custom_model_src."
                    )
                if not tb_task_id:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{benchmark_name} evaluation via /evaluate requires task_id or problem_id."
                    )

                result: KernelExecResult = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        eval_fn,
                        tb_kernel_code,
                        str(tb_task_id),
                        device,
                        runtime_timeout,
                        request.measure_performance,
                        request.num_perf_trials,
                    ),
                    timeout=runtime_timeout + 10
                )
            else:
                # Set up build directory (optional, can be configured)
                # Use build_dir from request, or fall back to environment variable
                build_dir = request.build_dir or os.environ.get("BUILD_DIR", None)
                if build_dir:
                    os.makedirs(build_dir, exist_ok=True)

                # Run evaluation in thread pool to avoid blocking event loop
                # The subprocess call itself is synchronous, so we run it in executor
                result: KernelExecResult = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        run_evaluation_sync,
                        request.original_model_src,
                        request.custom_model_src,
                        request.seed_num,
                        request.num_correct_trials,
                        request.num_perf_trials,
                        request.verbose,
                        request.measure_performance,
                        build_dir,
                        device,
                        request.backend,
                        request.dtype_str,
                        runtime_timeout,  # Pass runtime timeout to subprocess
                        request.gpu_name,
                        request.level,
                        request.problem_id,
                    ),
                    timeout=runtime_timeout + 10  # Add buffer for subprocess overhead
                )
            
            execution_time = time.time() - total_start_time
            
            # Handle None result (lock file errors, etc.)
            if result is None:
                return EvaluationResponse(
                    success=False,
                    compiled=False,
                    correctness=False,
                    execution_time=execution_time,
                    queue_time=queue_time,
                    error="Evaluation returned None (possible lock file error, please retry)"
                )
            
            # Convert result to response
            # Ensure metadata is JSON serializable
            metadata = result.metadata.copy()
            for key, value in metadata.items():
                if not isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    metadata[key] = str(value)
            
            return EvaluationResponse(
                success=result.correctness if is_tbg_eval else True,
                compiled=result.compiled,
                correctness=result.correctness,
                runtime=result.runtime,
                runtime_stats=result.runtime_stats,
                metadata=metadata,
                execution_time=execution_time,
                queue_time=queue_time
            )
            
        finally:
            # Always release the device slot
            if device_acquired:
                await resource_manager.release_device(device)
                device_acquired = False
        
    except asyncio.TimeoutError:
        execution_time = time.time() - total_start_time
        return EvaluationResponse(
            success=False,
            compiled=False,
            correctness=False,
            execution_time=execution_time,
            queue_time=queue_time,
            error=f"Kernel execution timeout: exceeded {runtime_timeout}s runtime limit"
        )
    except HTTPException:
        raise
    except AssertionError as e:
        execution_time = time.time() - total_start_time
        if device_acquired:
            await resource_manager.release_device(device)
        raise HTTPException(
            status_code=400,
            detail=f"Assertion error: {str(e)}"
        )
    except Exception as e:
        execution_time = time.time() - total_start_time
        if device_acquired:
            await resource_manager.release_device(device)
        
        error_msg = f"Evaluation failed: {str(e)}"
        
        # Log full error for debugging
        logger.error(f"ERROR: {error_msg}\n{traceback.format_exc()}")
        
        return EvaluationResponse(
            success=False,
            compiled=False,
            correctness=False,
            execution_time=execution_time,
            queue_time=queue_time if 'queue_time' in locals() else 0.0,
            error=str(e)  # Return user-friendly error message
        )


@app.post("/evaluate_fit_kernel")
async def evaluate_fit_kernel(request: FITEvaluationRequest):
    """
    Evaluate a FlashInfer Trace (FIT) kernel using flashinfer-bench

    This endpoint:
    1. Queues the request if all GPU slots are busy
    2. Creates a temporary solution file from the kernel code
    3. Runs flashinfer-bench to evaluate the kernel
    4. Returns the evaluation results
    """
    total_start_time = time.time()
    queue_time = 0.0
    device = None
    device_acquired = False  # Track if device was successfully acquired

    if DEBUG_MODE:
        logger.debug(f"Received FIT evaluation request: task_id={request.task_id}, backend={request.backend}, "
                    f"timeout={request.timeout}, device={request.device}")

    try:
        # Validate CUDA availability
        if not torch.cuda.is_available():
            raise HTTPException(
                status_code=503,
                detail="CUDA is not available. Cannot evaluate kernels."
            )

        # Determine device based on allocation mode
        if GPU_ALLOCATION_MODE == "auto":
            # Auto mode: automatically assign to least busy GPU, ignore request.device
            device = resource_manager.get_least_busy_device()
            if DEBUG_MODE:
                logger.debug(f"Auto allocation mode: assigned to device {device} (least busy)")
        else:
            # Manual mode: use specified device or default
            device = request.device
            if device is None:
                device = torch.cuda.current_device()

            # Validate device
            if device >= torch.cuda.device_count():
                error_msg = f"Device {device} not available. Only {torch.cuda.device_count()} device(s) available."
                logger.warning(f"400 Bad Request: {error_msg}")
                raise HTTPException(
                    status_code=400,
                    detail=error_msg
                )

            # Check if device is in the allowed list
            if not resource_manager.is_device_available(device):
                available_list = resource_manager.available_gpus if resource_manager.available_gpus else list(range(torch.cuda.device_count()))
                error_msg = f"Device {device} is not available. Available GPUs: {available_list}"
                logger.warning(f"400 Bad Request: {error_msg}")
                raise HTTPException(
                    status_code=400,
                    detail=error_msg
                )

        # Wait for device availability (with queue management)
        # queue_timeout: how long to wait for a device (use request.timeout directly, can be long)
        # runtime_timeout: max evaluation runtime after acquiring device (capped at RUNTIME_TIMEOUT)
        queue_timeout = request.timeout or RUNTIME_TIMEOUT
        runtime_timeout = min(request.timeout or RUNTIME_TIMEOUT, RUNTIME_TIMEOUT)
        queue_time = await resource_manager.wait_for_device(device, timeout=queue_timeout)
        device_acquired = True  # Mark device as successfully acquired

        # First try block: Create and save solution file
        solution_file = None
        solution_id = None
        op_type = None
        
        try:
            # Find op_type by scanning definition directories for the task_id file
            problem_name = request.task_id
            dataset_root = None
            definition_path = None

            for root in FLASHINFER_STYLE_DATASET_ROOTS:
                for candidate_op_type in FLASHINFER_TRACE_OP_TYPES:
                    candidate = os.path.join(root, "definitions", candidate_op_type, f"{request.task_id}.json")
                    if os.path.exists(candidate):
                        op_type = candidate_op_type
                        dataset_root = root
                        definition_path = candidate
                        break
                if op_type is not None:
                    break

            if op_type is None or dataset_root is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Definition not found for task_id={request.task_id} in any op_type directory of {FLASHINFER_TRACE_OP_TYPES}"
                )

            # Create temporary solution file
            import uuid
            solution_id = f"temp_{uuid.uuid4().hex[:8]}"
            solution_dir = os.path.join(dataset_root, "solutions", op_type, request.task_id)
            os.makedirs(solution_dir, exist_ok=True)

            solution_file = os.path.join(solution_dir, f"{solution_id}.json")

            # Create solution JSON
            solution_data = {
                "name": solution_id,
                "definition": request.task_id,
                "author": "kernelbench_eval",
                "spec": {
                    "language": request.backend,
                    "target_hardware": ["B200"],  # Default target
                    "entry_point": "main.py::run",
                    "dependencies": [],
                    "destination_passing_style": False
                },
                "sources": [
                    {
                        "path": "main.py",
                        "content": request.kernel_code
                    }
                ],
                "description": f"KernelBench evaluation for {request.task_id}"
            }

            # Write solution file
            with open(solution_file, 'w') as f:
                json.dump(solution_data, f, indent=2)
            
            print(f"Solution file created: {solution_file}")

        except Exception as e:
            
            # return error message
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create solution file: {str(e)}"
            )

        # Second try block: Run evaluation
        try:
            # Run flashinfer-bench with CUDA_VISIBLE_DEVICES set for the allocated device
            cmd = [
                "flashinfer-bench", "run",
                "--local", dataset_root,
                "--definitions", request.task_id,
                "--solutions", solution_id,
                "--timeout", str(60),
                "--warmup-runs", str(3),
                "--num-trials", str(1),
                "--iterations", str(5)
            ]

            print(f"Running command: {cmd}")

            if DEBUG_MODE:
                logger.debug(f"Running command on device {device}: {cmd}")

            # Set up environment with CUDA_VISIBLE_DEVICES for the allocated device
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(device)

            # Record the time before flashinfer-bench started to find new results
            run_start_time = time.time()

            # Set timeout and run
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=REPO_TOP_PATH,
                env=env
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=runtime_timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                raise HTTPException(
                    status_code=408,
                    detail=f"FlashInfer evaluation timeout: exceeded {runtime_timeout}s"
                )

            # Check return code
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise HTTPException(
                    status_code=500,
                    detail=f"FlashInfer evaluation failed: {error_msg}"
                )

            # Read results from jsonl file instead of stdout
            traces_file = os.path.join(dataset_root, "traces", op_type, f"{request.task_id}.jsonl")
            os.makedirs(os.path.dirname(traces_file), exist_ok=True)

            if not os.path.exists(traces_file):
                raise HTTPException(
                    status_code=500,
                    detail=f"Traces file not found: {traces_file}"
                )

            # Collect all results for our solution that were created after we started
            evaluation_results = []

            with open(traces_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        result = json.loads(line.strip())
                        if (result.get("solution") == solution_id and
                            result.get("definition") == request.task_id):

                            # Check timestamp to ensure this result is from our run
                            timestamp_str = result.get("evaluation", {}).get("timestamp")
                            if timestamp_str:
                                # Parse timestamp
                                from datetime import datetime
                                result_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                result_time = result_timestamp.timestamp()

                                # Only consider results created after we started the evaluation
                                if result_time >= run_start_time:
                                    evaluation_results.append(result)
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines

            if not evaluation_results:
                raise HTTPException(
                    status_code=500,
                    detail=f"No evaluation results found for solution {solution_id} in traces file (searched from {run_start_time})"
                )

            # Aggregate results across all workloads
            total_workloads = len(evaluation_results)

            # Define error statuses that should return log
            error_statuses = {
                "COMPILE_ERROR",
                "RUNTIME_ERROR",
                "INCORRECT_SHAPE",
                "INCORRECT_NUMERICAL",
                "INCORRECT_DTYPE"
            }

            # Check for errors first - find first error result
            first_error_result = None
            for result in evaluation_results:
                status = result.get("evaluation", {}).get("status", "")
                if status in error_statuses:
                    first_error_result = result
                    break

            # If there's an error, return the log from the first error
            if first_error_result is not None:
                error_log = first_error_result.get("evaluation", {}).get("log", "No log available")
                error_status = first_error_result.get("evaluation", {}).get("status", "UNKNOWN_ERROR")
                
                execution_time = time.time() - total_start_time
                
                return EvaluationResponse(
                    success=False,
                    compiled=(error_status != "COMPILE_ERROR"),
                    correctness=False,
                    runtime=-1.0,
                    runtime_stats={},
                    metadata={"task_id": request.task_id, "backend": request.backend, "device": device, "error_status": error_status, "error_log": error_log},
                    execution_time=execution_time,
                    queue_time=queue_time
                )

            # No errors - proceed with normal processing
            # Since we've filtered out all error statuses, all remaining results should be PASSED
            all_passed = True  # All results are PASSED if we reach here
            any_compiled = True  # PASSED means compilation succeeded
            all_correct = True  # All results are correct if we reach here (no errors)

            # Calculate average performance metrics
            latencies = []
            reference_latencies = []
            speedup_factors = []
            max_relative_errors = []
            max_absolute_errors = []

            for result in evaluation_results:
                performance = result.get("evaluation", {}).get("performance", {})
                correctness = result.get("evaluation", {}).get("correctness", {})

                if "latency_ms" in performance:
                    latencies.append(performance["latency_ms"])
                if "reference_latency_ms" in performance:
                    reference_latencies.append(performance["reference_latency_ms"])
                if "speedup_factor" in performance:
                    speedup_factors.append(performance["speedup_factor"])
                if "max_relative_error" in correctness:
                    max_relative_errors.append(correctness["max_relative_error"])
                if "max_absolute_error" in correctness:
                    max_absolute_errors.append(correctness["max_absolute_error"])

            # Compute averages
            avg_runtime = sum(latencies) / len(latencies) if latencies else -1.0
            avg_reference_latency = sum(reference_latencies) / len(reference_latencies) if reference_latencies else -1.0
            avg_speedup_factor = sum(speedup_factors) / len(speedup_factors) if speedup_factors else 1.0
            avg_max_relative_error = sum(max_relative_errors) / len(max_relative_errors) if max_relative_errors else -1.0
            avg_max_absolute_error = sum(max_absolute_errors) / len(max_absolute_errors) if max_absolute_errors else -1.0

            # Map to our response format
            success = all_passed
            compiled = any_compiled
            correctness = all_correct

            runtime = avg_runtime  # Average latency in milliseconds
            runtime_stats = {
                "reference_latency_ms": avg_reference_latency,
                "speedup_factor": avg_speedup_factor,
                "max_relative_error": avg_max_relative_error,
                "max_absolute_error": avg_max_absolute_error,
                "total_workloads": total_workloads,
                "fast_p": avg_speedup_factor
            }

            execution_time = time.time() - total_start_time

            return EvaluationResponse(
                success=success,
                compiled=compiled,
                correctness=correctness,
                runtime=runtime,
                runtime_stats=runtime_stats,
                metadata={"task_id": request.task_id, "backend": request.backend, "device": device},
                execution_time=execution_time,
                queue_time=queue_time
            )
        
        except HTTPException:
            # Re-raise HTTPException so FastAPI can handle it properly
            raise

        except Exception as e:
            execution_time = time.time() - total_start_time
            logger.error(f"Unexpected error in FIT evaluation: {traceback.format_exc()}")
            return EvaluationResponse(
                success=False,
                compiled=False,
                correctness=False,
                execution_time=execution_time,
                queue_time=queue_time,
                error=str(e)
            )

        finally:
            # Ensure the solution file is removed
            if solution_file and os.path.exists(solution_file):
                pass
                # try:
                #     os.remove(solution_file)
                # except Exception as e:
                #     logger.warning(f"Failed to remove solution file {solution_file}: {e}")

    except asyncio.TimeoutError:
        execution_time = time.time() - total_start_time
        return EvaluationResponse(
            success=False,
            compiled=False,
            correctness=False,
            execution_time=execution_time,
            queue_time=queue_time,
            error=f"Kernel execution timeout: exceeded {runtime_timeout}s runtime limit"
        )
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - total_start_time
        logger.error(f"Unexpected error in FIT evaluation: {traceback.format_exc()}")
        return EvaluationResponse(
            success=False,
            compiled=False,
            correctness=False,
            execution_time=execution_time,
            queue_time=queue_time,
            error=str(e)
        )

    finally:    
        # Always release the device slot if it was successfully acquired
        if device_acquired and device is not None:
            await resource_manager.release_device(device)


async def _evaluate_tritonbench_kernel(request, *, benchmark: str, eval_fn) -> EvaluationResponse:
    """
    Evaluate a TritonBench kernel with official built-in tests.
    """
    benchmark = str(benchmark).upper()
    total_start_time = time.time()
    queue_time = 0.0
    device = None
    device_acquired = False
    # queue_timeout: how long to wait for a device (use request.timeout directly, can be long)
    # runtime_timeout: max evaluation runtime after acquiring device (capped at RUNTIME_TIMEOUT)
    queue_timeout = request.timeout or RUNTIME_TIMEOUT
    runtime_timeout = min(request.timeout or RUNTIME_TIMEOUT, RUNTIME_TIMEOUT)

    if DEBUG_MODE:
        logger.debug(
            f"Received {benchmark} evaluation request: task_id={request.task_id}, backend={request.backend}, "
            f"timeout={request.timeout}, device={request.device}"
        )

    try:
        # Validate CUDA availability
        if not torch.cuda.is_available():
            raise HTTPException(
                status_code=503,
                detail="CUDA is not available. Cannot evaluate kernels."
            )

        # Determine device based on allocation mode
        if GPU_ALLOCATION_MODE == "auto":
            device = resource_manager.get_least_busy_device()
            if DEBUG_MODE:
                logger.debug(f"Auto allocation mode: assigned to device {device} (least busy)")
        else:
            device = request.device
            if device is None:
                device = torch.cuda.current_device()

            if device >= torch.cuda.device_count():
                error_msg = f"Device {device} not available. Only {torch.cuda.device_count()} device(s) available."
                logger.warning(f"400 Bad Request: {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)

            if not resource_manager.is_device_available(device):
                available_list = resource_manager.available_gpus if resource_manager.available_gpus else list(range(torch.cuda.device_count()))
                error_msg = f"Device {device} is not available. Available GPUs: {available_list}"
                logger.warning(f"400 Bad Request: {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)

        queue_time = await resource_manager.wait_for_device(device, timeout=queue_timeout)
        device_acquired = True

        loop = asyncio.get_event_loop()
        result: KernelExecResult = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                eval_fn,
                request.kernel_code,
                request.task_id,
                device,
                runtime_timeout,
                request.measure_performance,
                request.num_perf_trials,
            ),
            timeout=runtime_timeout + 10,
        )

        execution_time = time.time() - total_start_time
        metadata = result.metadata.copy()
        for key, value in metadata.items():
            if not isinstance(value, (dict, list, str, int, float, bool, type(None))):
                metadata[key] = str(value)

        return EvaluationResponse(
            success=result.correctness,
            compiled=result.compiled,
            correctness=result.correctness,
            runtime=result.runtime,
            runtime_stats=result.runtime_stats,
            metadata=metadata,
            execution_time=execution_time,
            queue_time=queue_time,
        )

    except asyncio.TimeoutError:
        execution_time = time.time() - total_start_time
        return EvaluationResponse(
            success=False,
            compiled=False,
            correctness=False,
            execution_time=execution_time,
            queue_time=queue_time,
            error=f"Kernel execution timeout: exceeded {runtime_timeout}s runtime limit"
        )
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - total_start_time
        logger.error(f"Unexpected error in {benchmark} evaluation: {traceback.format_exc()}")
        return EvaluationResponse(
            success=False,
            compiled=False,
            correctness=False,
            execution_time=execution_time,
            queue_time=queue_time,
            error=str(e),
        )
    finally:
        if device_acquired and device is not None:
            await resource_manager.release_device(device)


@app.post("/evaluate_tbg_kernel", response_model=EvaluationResponse)
async def evaluate_tbg_kernel(request: TBGEvaluationRequest):
    """
    Evaluate a TritonBench-G kernel using built-in task tests.

    When `measure_performance=True`, the judge also benchmarks the official test
    function and returns `runtime_stats.fast_p`.
    """
    return await _evaluate_tritonbench_kernel(
        request,
        benchmark="TBG",
        eval_fn=run_tbg_evaluation_sync,
    )


@app.post("/evaluate_tbt_kernel", response_model=EvaluationResponse)
async def evaluate_tbt_kernel(request: TBTEvaluationRequest):
    """
    Evaluate a TritonBench-T kernel using built-in task tests.

    When `measure_performance=True`, the judge also benchmarks the official test
    function and returns `runtime_stats.fast_p`.
    """
    return await _evaluate_tritonbench_kernel(
        request,
        benchmark="TBT",
        eval_fn=run_tbt_evaluation_sync,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(
        "app_with_queue:app",
        host=host,
        port=port,
        reload=os.environ.get("DEBUG", "false").lower() == "true",
        log_level="warning"
    )
