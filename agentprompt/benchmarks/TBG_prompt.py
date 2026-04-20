TBG_TRITON_PROMPT = """Generate a Python module for TritonBench-G task `{task_id}`.

Reference task file (contains reference implementation and built-in tests):

```python
{arc_src}
```

Target GPU: {target_gpu}

Requirements:
- Correctness is the top priority. The main metric is execution accuracy.
- Keep function/wrapper names compatible with the task tests.
- Include required imports (torch, triton, triton.language as tl) when needed.
- You may use Triton kernels and wrappers, but avoid unnecessary complexity.
- Output only runnable Python code (no markdown, no explanations).
- Do not include extra custom test runner code.
"""

