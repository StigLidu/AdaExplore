# Released AdaExplore kernels

`kernelbench/level2/<problem_id>.py` and `kernelbench/level3/<problem_id>.py` hold the best kernel
AdaExplore found for each KernelBench problem. `<problem_id>` is the numeric prefix of the
KernelBench problem file (natural order, i.e. `25.py` ↔ `25_Conv2d_Min_Tanh_Tanh.py`), matching
`construct_kernelbench_dataset`.

## Provenance

| | source run | checkpoint |
|---|---|---|
| Level 2 (100) | `experiments/release/MCTS_0120_KB-l2_200` | `global_best_kernel_200.py` |
| Level 3 (50) | `experiments/release/MCTS_0120_KB-l3_100` | `global_best_kernel_100.py` |

Per-problem provenance and the evaluation results recorded during those runs are in
`kernelbench/metrics_level2.json` / `metrics_level3.json`. Five problems (L2 47, L3 5/14/33/44)
come from that run's `*_BUG` directory, which is the only directory present for them — the same
fallback `tool_scripts/re_evaluate.py` applies.

## Caveats on the archived metrics

The metrics are **copied verbatim from the runs; nothing was re-measured when archiving.** So:

- **Correctness was judged at `torch.allclose(atol=5e-02, rtol=5e-02)`**, the gate in force when
  these runs were produced. That is looser than KernelBench v0/v0.1 (1e-2) and much looser than
  current KernelBench main (1e-4 for fp32). `src/eval.py` now uses 1e-2; these kernels have not
  been re-checked at that gate. See `tool_scripts/rerun_at_1e-2.md`.
- `recorded_baseline` computes `fast_p` against the fixed `results/timing/A6000/baseline_time_torch.json`;
  `measured_baseline` re-times the reference in the same process, and is noisier. They disagree by
  a few percent on most problems and by more on short kernels.
- fp32, TF32 disabled, NVIDIA RTX A6000.

## Previous version

Before this update the directory held the kernels from the 50-step runs
`MCTS-PDR-FB-P4_0120_KB-l2-1-25_v1_50_260314-005531` (L2) and
`CHAIN-P4-new_0120_KB-l3_v1_50_260324-005531` (L3), both `global_best_kernel_50.py`. They remain in
git history. Against those, under the same 5e-2 evaluation and using the recorded baselines, the
kernels archived here are faster in aggregate:

| | mean speedup | geometric mean | median | Fast@1.2 | Fast@2.0 |
|---|---|---|---|---|---|
| Level 2 | 2.63 → 3.76 | 1.93 → 2.83 | 1.41 → 2.54 | 0.67 → 0.86 | 0.30 → 0.56 |
| Level 3 | 1.57 → 1.97 | 1.29 → 1.60 | 1.09 → 1.26 | 0.32 → 0.62 | 0.18 → 0.22 |

Speedups clamped at 10.0, as in `tool_scripts/stats.py`. The aggregate improvement is not uniform
per problem: 16 Level-2 and 7 Level-3 problems are slower than the previous version (33 and 17
under the measured baseline).
