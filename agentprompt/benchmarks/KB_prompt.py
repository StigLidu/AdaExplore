KB_TRITON_PROMPT = """
## Hardware Information

Here is some information about the underlying hardware that you should keep in mind:

- The GPU that will run the kernel is NVIDIA {gpu_name}, {gpu_architecture} architecture.

## Task Instruction

You are given the following architecture:

```python
{arc_src}
```

The input shapes can be found in the input of the architecture, and the dtype is {dtype_str}.

Optimize the architecture named Model with custom Triton kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!
"""