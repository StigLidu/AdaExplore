# Released AdaExplore kernels

`kernelbench/level2/<problem_id>.py` and `kernelbench/level3/<problem_id>.py` hold the best kernel
AdaExplore found for each KernelBench problem. `<problem_id>` is the numeric prefix of the
KernelBench problem file (natural order, i.e. `25.py` ↔ `25_Conv2d_Min_Tanh_Tanh.py`), matching
`construct_kernelbench_dataset`.
