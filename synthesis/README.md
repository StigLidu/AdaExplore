### Synthesis (synthetic KernelBench data)

This folder contains scripts to **generate** synthetic PyTorch kernel modules with an LLM and then **materialize** them into a KernelBench-style dataset folder (e.g. `datasets/KernelBench_syn/syn_v4`) **sequentially**.

### Recommended sequential workflow

- **Step 1 — Generate kernels (sequential)**: `synthesis/generate_data.py`
- **Step 2 — Copy + renumber dataset files (sequential)**: `synthesis/rename.py`

### Step 1: Generate synthetic kernels

Run from the repo root:

```bash
python synthesis/generate_data.py \
  --server_type azure \
  --model_name gpt-5-mini \
  --prompt_style composite \
  --input_levels 1 \
  --num_generations 200 \
  --num_examples_per_request 3 \
  --temperature 1.0
```

- **Output**: a new directory under `outputs/data_generation/` named like:
  - `generated_data_<prompt_style>_<model>_<YYYYMMDD_HHMMSS>/`
- **Validation**:
  - By default, generated kernels are execution-validated on **CUDA** (GPU required).
  - Use `--skip_validation` if you only want a basic structural check.
  - By default validation runs in a **subprocess** (safer for GPU/runtime crashes). You can disable that with `--no_subprocess_validation`.

### Step 2: Copy outputs into a dataset folder and renumber `*.py`

After Step 1 finishes, copy the generated directory into your dataset path and renumber Python files to `1.py, 2.py, ...`:

```bash
python synthesis/rename.py \
  --source_path outputs/data_generation/generated_data_composite_gpt-5-mini_YYYYMMDD_HHMMSS \
  --data_path datasets/KernelBench_syn/syn_v4 \
  --force
```

- **What it does**:
  - Deletes `--data_path` first if it exists and `--force` is provided
  - Copies `--source_path` to `--data_path`
  - Removes non-`.py` artifacts in the destination (e.g. prompt/response `.txt`)
  - Renames `.py` files in natural-sorted order to `1.py, 2.py, ...`

### Notes

- **Sequential execution**: the generation loop and the copy/renumber step are both designed to run **sequentially** (run Step 1, then Step 2).
- `synthesis/valid_subprocess_runner.py` is an internal helper used by `generate_data.py` for isolated validation.

