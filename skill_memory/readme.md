## How to collect the skill memory

This repo uses a skill memory:

- **Skill memory (experience guidance)**: *log-grounded constraints* distilled from **failed** kernels (compile/runtime errors).  
  Format is a line-based memory file: `You cannot ...||<score (frequency)>`.

Below is the end-to-end guidance to collect and use it.

---

## Prerequisites: what logs look like

Both collectors assume you already have an agent run under `outputs/...`, with per-problem folders containing step artifacts like:

- `step_<k>.py`: generated kernel code at step \(k\)
- `step_<k>_metrics.(txt|json)`: evaluation metrics (compiled/correctness/runtime/speedup, etc.)
- `step_<k>_prompt.txt`: the prompt used at step \(k\) (optional but recommended)

Example:

- `outputs/example_run/2_1/step_7.py`
- `outputs/example_run/2_1/step_7_metrics.json`
- `outputs/example_run/2_1/step_7_prompt.txt`

---

## Skill memory collection

### What it is

Skill memory is extracted from **error logs only** (CE/RE-type failures where `correctness=False`).  
The extractor asks an LLM to produce **exactly one** minimal rule:

- Must start with: `You cannot ...`
- Must be strictly grounded in the error message
- If unclear, it outputs `no guidance` and will be skipped

The collected memory is stored in a line-based memory file (for example under `results/memory/`), where each line is:

```
<knowledge_item>||<score>
```

The `<score>` is a simple frequency-like counter; it increases when the new item is judged as a duplicate of an existing one (LLM-based duplicate judge in `skill_memory/deduplicate_knowledge.py`).

### How it is used by the agent

At runtime, the proposer prompt optionally injects items whose score is above a threshold:

- Memory file path: `--general_memory_path`
- Enable updating (write-back): `--memory_update`
- Injection threshold (in proposer prompt): `--knowledge_1_threshold` (default: `3`)

So, **collect many**, then **control how much gets injected** with the threshold.

```bash
  --filter-max-difference
```

### Option A: update skill memory during the agent run (online update, recommended)

When running the agent via `agent/agent_entry.py`, you can enable auto-update:

```bash
python agent/agent_entry.py \
  --config config/<your_config>.yaml \
  --general_memory_path results/memory/general_memory_v1_200.txt \
  --memory_update \
  --knowledge_1_threshold 3
```

Notes:

- This path uses a file lock (`<memory>.lock`) to avoid concurrent write corruption.
- The update happens after a problem finishes, by calling `skill_memory/skill_memory.update_memory(...)` on that problem’s log folder.

### Option B: collect skill memory from an existing `outputs/<run>/<problem>/...` folder

If you have a specific run folder (e.g. `outputs/example_run`) and want to update a memory file from the errors inside it, you can run:

```bash
python skill_memory/skill_memory.py \
  --log-dir outputs/example_run \
  --knowledge-store-path results/memory/general_memory_v1_200.txt \
  --server azure \
  --model-name gpt-5-mini \
  --max-logs 3000 \
  --seed 42
```

Notes:

- It scans `outputs/<run>/*/*_metrics.(txt|json)` and only keeps logs where **`correctness=False`**.
- To drop “max_difference” related cases (i.e., correctness issue cases, often noisy), add:

---

## Quick sanity checks

- **Skill memory**: open your memory file (for example `results/memory/general_memory_v1_200.txt`) and confirm each line looks like:
  - starts with `You cannot ...`
  - ends with `||<number>`