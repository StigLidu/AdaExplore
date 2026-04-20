"""
KernelBench NCU runner - standalone script for profiling solutions.

This module is intended to be invoked under NVIDIA Nsight Compute (ncu), e.g.:

    ncu --nvtx --nvtx-include kernelbench_ncu_profile -f \
      python -u -m src.ncu_subprocess_runner --data-dir /path/to/payload --device cuda:0

The payload directory is expected to contain an `args.json` with (at least):
  - original_model_src: str
  - custom_model_src: str

Optional keys:
  - seed_num: int
  - backend: str ("cuda" | "triton")
  - dtype_str: str ("fp16" | "fp32" | "bf16")
  - build_dir: str | None
  - num_warmup: int
  - nvtx_range: str
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Configure logging to stderr (avoid mixing with tool stdout as much as possible)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def _ensure_repo_root_on_syspath() -> None:
    """Ensure repo root is importable for `import src.*` when executed as a script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # If this file is in <repo>/src/, add <repo> to sys.path
    if os.path.basename(script_dir) == "src":
        repo_root = os.path.dirname(script_dir)
    else:
        repo_root = script_dir
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _parse_device(device: Union[str, int]) -> Tuple[str, Optional[int]]:
    """
    Normalize device into (device_str, device_index).

    Accepts:
      - "cuda:0", "cuda", "0"
      - int 0
    """
    if isinstance(device, int):
        return f"cuda:{device}", device

    s = str(device).strip()
    if s.startswith("cuda:"):
        try:
            return s, int(s.split("cuda:")[1])
        except Exception:
            return s, None
    if s == "cuda":
        return "cuda", None
    # treat numeric strings as cuda indices
    if s.isdigit():
        idx = int(s)
        return f"cuda:{idx}", idx
    return s, None


def _read_payload_args(data_dir: Path) -> Dict[str, Any]:
    args_path = data_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Payload args.json not found: {args_path}")
    return json.loads(args_path.read_text())


def _cast_inputs_dtype(inputs, dtype_str: str, device_index: Optional[int]):
    import torch

    if dtype_str == "fp16":
        dtype = torch.float16
    elif dtype_str == "fp32":
        dtype = torch.float32
    elif dtype_str == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Invalid dtype_str: {dtype_str}")
    return [
        x.cuda(device=device_index).to(dtype=dtype) if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]


def main() -> None:
    _ensure_repo_root_on_syspath()

    parser = argparse.ArgumentParser(description="Run KernelBench solution for NCU profiling")
    parser.add_argument("--data-dir", required=True, help="Path to payload directory")
    parser.add_argument("--device", default="cuda:0", help="CUDA device, e.g. cuda:0")
    parser.add_argument("--backend", default=None, help="Backend hint: cuda|triton (overrides args.json)")
    parser.add_argument("--dtype-str", default=None, help="DType: fp16|fp32|bf16 (overrides args.json)")
    parser.add_argument("--seed", type=int, default=None, help="Seed (overrides args.json)")
    parser.add_argument("--build-dir", default=None, help="Build dir for CUDA extensions (overrides args.json)")
    parser.add_argument("--num-warmup", type=int, default=None, help="Warmup runs (overrides args.json)")
    parser.add_argument(
        "--nvtx-range",
        default=None,
        help="NVTX range name used by ncu --nvtx-include (overrides args.json)",
    )
    args = parser.parse_args()

    import torch

    data_dir = Path(args.data_dir)
    payload = _read_payload_args(data_dir)

    original_model_src_path = payload["original_model_src_path"]
    custom_model_src_path = payload["custom_model_src_path"]

    seed_num = args.seed if args.seed is not None else int(payload.get("seed_num", 42))
    backend = args.backend if args.backend is not None else str(payload.get("backend", "cuda"))
    dtype_str = args.dtype_str if args.dtype_str is not None else str(payload.get("dtype_str", "fp32"))
    build_dir = args.build_dir if args.build_dir is not None else payload.get("build_dir")
    num_warmup = args.num_warmup if args.num_warmup is not None else int(payload.get("num_warmup", 1))
    nvtx_range = args.nvtx_range if args.nvtx_range is not None else str(payload.get("nvtx_range", "kernelbench_ncu_profile"))

    device_str, device_index = _parse_device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; NCU profiling requires a CUDA-enabled environment.")
    if device_index is None:
        # Default to current device if not parseable
        device_index = torch.cuda.current_device()
        device_str = f"cuda:{device_index}"

    # Triton kernels can be sensitive to CUDA_VISIBLE_DEVICES; mimic eval.py behavior.
    # Must set CUDA_VISIBLE_DEVICES *before* set_device, because restricting
    # visible devices remaps the physical GPU to index 0.
    if backend == "triton":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
        device_index = 0
        device_str = "cuda:0"

    torch.cuda.set_device(device_index)

    # Import after sys.path fix
    from src.eval import (
        graceful_eval_cleanup,
        load_custom_model_from_file,
        load_custom_model_with_tempfile_from_file,
        load_original_model_and_inputs_from_file,
        set_seed,
    )

    logger.info("NCU runner starting: device=%s backend=%s dtype=%s warmup=%s", device_str, backend, dtype_str, num_warmup)

    context: Dict[str, Any] = {}
    tempfile_obj = None

    try:
        # Load reference helpers (for deterministic inputs)
        Model, get_init_inputs_fn, get_inputs_fn = load_original_model_and_inputs_from_file(original_model_src_path, context)
        if Model is None or get_init_inputs_fn is None or get_inputs_fn is None:
            raise RuntimeError("Failed to load original model / input functions from original_model_src.")

        set_seed(seed_num)
        init_inputs = get_init_inputs_fn()
        init_inputs = [
            x.cuda(device=device_index) if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]

        # Compile / load custom model class
        set_seed(seed_num)
        if backend == "triton":
            ModelNew, tempfile_obj = load_custom_model_with_tempfile_from_file(
                custom_model_src_path, entry_point="ModelNew"
            )
        else:
            ModelNew = load_custom_model_from_file(
                custom_model_src_path, context, build_dir, entry_point="ModelNew"
            )
        torch.cuda.synchronize(device=device_index)
        if ModelNew is None:
            raise RuntimeError("Failed to load ModelNew from custom_model_src.")

        # Instantiate model
        with torch.no_grad():
            set_seed(seed_num)
            custom_model = ModelNew(*init_inputs)
        torch.cuda.synchronize(device=device_index)

        # Prepare inputs (once) and run warmups
        set_seed(seed_num)
        inputs = get_inputs_fn()
        inputs = _cast_inputs_dtype(inputs, dtype_str=dtype_str, device_index=device_index)
        model_new = custom_model.cuda(device=device_index)
        torch.cuda.synchronize(device=device_index)

        with torch.no_grad():
            for _ in range(max(0, num_warmup)):
                model_new(*inputs)
                torch.cuda.synchronize(device=device_index)

            # Actual run for profiling (filterable by NVTX range)
            with torch.cuda.nvtx.range(nvtx_range):
                model_new(*inputs)
                torch.cuda.synchronize(device=device_index)

    finally:
        # Try to cleanup resources (tempfiles, context, etc.)
        try:
            from src.eval import graceful_eval_cleanup as _cleanup

            _cleanup(context, device_index, tempfile_obj)
        except Exception:
            # Best-effort cleanup only; avoid masking upstream exception.
            pass


if __name__ == "__main__":
    main()

