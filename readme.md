# AdaExplore

HardAgent is a research codebase for **LLM-driven GPU kernel engineering**. It combines:

- a multi-strategy search agent for proposing and revising kernels
- a benchmark and evaluation harness for correctness and performance
- optional memory components for iterative improvement
- an optional remote evaluation service for multi-GPU or cluster setups

The repository is built around KernelBench-style tasks, but also includes support for synthesized tasks, FlashInfer-style traces, MLSYS-style tasks, and TritonBench-G style inputs.

## What It Does

Given a reference PyTorch operator or model component, HardAgent asks an LLM to generate optimized CUDA/Triton-style implementations, evaluates the generated kernels, and uses search to keep improving them.

The main supported agent families are:

- `MCTS`: tree search with proposal and revision steps
- `IRS`: iterative refinement with small-step revisions
- `IRL`: iterative refinement with large-step proposals
- `PS`: parallel sampling

## Repository Structure

```text
.
├── agent/               # Main search agents and entrypoints
├── agentprompt/         # Prompt templates
├── config/              # YAML configs for experiments and runs
├── datasets/            # Local task datasets
├── online_judge/        # Optional remote evaluation service
├── results/             # Saved kernels, memory files, and evaluation artifacts
├── tool_scripts/        # Optional evaluation / analysis helpers
├── src/                 # Evaluation harness and dataset loading
├── skill_memory/        # Knowledge and memory update utilities
├── synthesis/           # Synthetic task generation utilities
```

## Installation

This project expects a Python environment with GPU support.

```bash
conda create -n AdaExplore python=3.10
conda activate AdaExplore
pip install -r requirements.txt
```

Depending on your setup, you may also need keys such as `ANTHROPIC_API_KEY`, in addition to the backend-specific variables used by `agent/inference_server.py`.

## Quick Start

### 1. Start the evaluation service

Many configs assume remote evaluation. For a local single-machine setup, you can start the bundled judge service:

```bash
python -m uvicorn online_judge.app_with_queue:app --host 0.0.0.0 --port 12017
```

More deployment options are documented in `online_judge/README.md`.

### 2. Run the agent

The main entrypoint is:

```bash
python agent/agent_entry.py --config <CONFIG_PATH>
```

Example:

```bash
python agent/agent_entry.py --config config/KB-l2/config_KB-l2_AdaExplore_50.yaml
```

Other sample configs currently kept in the repo:

- `config/KB-l3/config_KB-l3_AdaExplore_50.yaml`
- `config/SYN-v4/config_SYN-v4_none_MCTS.yaml`

## Important Runtime Options

`agent/agent_entry.py` is the main CLI. Common options include:

- `--test_source`: one of `KB`, `SYN`, `FIT`, `MLSYS`, `TBG`
- `--agent_type`: one of `IRS`, `IRL`, `IRLE`, `PS`, `MCTS`
- `--level` and `--problem_id`: task selection
- `--config`: load a YAML config
- `--use_remote_eval` and `--remote_eval_url`: use the online judge instead of local in-process evaluation
- `--server_type` and `--model_name`: choose the inference backend and model

## Datasets

The codebase supports multiple task sources:

- `KB`: KernelBench-style tasks under `datasets/KernelBench`
- `SYN`: synthesized tasks under `datasets/KernelBench_syn`
- `FIT`: FlashInfer-style trace tasks
- `MLSYS`: MLSYS contest-style tasks
- `TBG`: TritonBench-G style tasks

Dataset path resolution lives in `src/dataset.py`.

## Evaluation

The core evaluation logic is implemented in `src/eval.py`.

The harness checks:

- **correctness** against the reference implementation
- **performance** through repeated timing trials
- **combined metrics** such as `fast_p`, which measures the fraction of tasks that are both correct and faster than a chosen threshold

The repository also includes optional helper scripts in `tool_scripts/` for:

- evaluating one kernel against a task
- re-evaluating saved runs
- summarizing metrics
- generating baseline timings

## Additional Entry Points

- `tool_scripts/`: optional evaluation and analysis helpers
- `skill_memory/readme.md`: skill-memory collection and update workflow
- `synthesis/README.md`: synthetic task generation workflow
- `src/README.md`: lower-level benchmark and evaluation workflow
- `online_judge/README.md`: remote evaluation service details

## Notes on Configs

Configs in `config/` encode the dataset, search strategy, model backend, evaluation mode, and logging location. A typical config contains:

- the task source and level
- the agent type
- GPU and process settings
- whether remote evaluation is enabled
- model provider and model name
- search hyperparameters

## License

This repository is released under the MIT License. See `LICENSE`.