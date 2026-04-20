import re
from agentprompt.prompt_modules import generate_experience_guidance_prompt
from agentprompt.prompt_modules import generate_hardware_information_prompt
from src.utils import read_file
from src.eval import KernelExecResult
import os

def _extract_format_keys(template: str):
    """Extract format keys from a template string."""
    return set(re.findall(r'\{(\w+)\}', template))

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

PROBLEM_STATEMENT = """## Problem Statement

You revise the custom Triton kernels in the given architecture to get better performance. Beyond kernel-level tuning, also consider whether reordering mathematically equivalent operations could enable better fusion or memory access patterns.

"""

TASK_INSTRUCTION = """## Task Instruction

You are given the following architecture:

```
{arc_src}
```

The input shapes can be found in the input of the architecture, and the dtype is {dtype_str}.

An agent generated the following custom Triton kernels in the architecture:

```
{custom_triton_kernels}
```

The runtime metrics of the custom Triton kernels are:

```
{run_info}
```

The tuning metrics contain the following information:

* **Compiled**: whether the kernel is compiled successfully
* **Error Message**: the compilation or runtime error encountered by the kernel (if any)
* **Correctness**: whether the kernel is correct
* **Runtime**: the runtime of the kernel
* **Fast_p**: compared with the standard PyTorch implementation, how much speedup the customized kernel achieves, calculated as *standard time / custom time*.

### Goal

Your goal is to help the agent improve the performance of the custom Triton kernels, and correct the correctness errors if any. Improvements may include not only low-level kernel optimizations, but also restructuring the computation order when mathematically equivalent (e.g., reordering elementwise ops, folding normalization into linear layers) to enable better fusion or memory access.

Output a concise guidance to the agent on how to improve the performance of the custom Triton kernels, remember:
- The revise is iterative, so the guidance should be concise and only contain 1-3 most important improvements.

"""

HARDWARE_INFORMATION = """## Hardware Information

Here is some information about the underlying hardware that you should keep in mind:

- The GPU that will run the kernel is NVIDIA {gpu_name}, {gpu_architecture} architecture.

"""

EXPERIENCE_GUIDANCE = """## Experience Guidance

Here is some experience guidance that you should keep in mind:

{experience_guidance}

"""

def generate_reviser_prompt(custom_triton_kernels: str=None, run_info: str=None, experience_guidance_path: str=None, task_params: dict=None, knowledge_1_threshold: int=3):
    # Extract required parameters from task prompt template
    required_keys = _extract_format_keys(TASK_INSTRUCTION)
    
    # Build format dict: use task_params if provided, otherwise fall back to original parameters
    format_dict = {}
    if task_params is not None:
        format_dict.update(task_params)
    
    # Fall back to original parameters for missing keys
    for key in required_keys:
        if key not in format_dict:
            if key == 'custom_triton_kernels':
                format_dict[key] = custom_triton_kernels
            elif key == 'run_info':
                format_dict[key] = run_info
            else:
                raise ValueError(f"Missing required parameter: {key}")
    
    prompt = PROBLEM_STATEMENT
    prompt += generate_experience_guidance_prompt(experience_guidance_path, threshold=knowledge_1_threshold)
    prompt += generate_hardware_information_prompt(task_params.get('gpu_name'), task_params.get('gpu_architecture'))
    prompt += TASK_INSTRUCTION.format(**format_dict)
    return prompt

if __name__ == "__main__":
    EXAMPLE_ARCH_SRC = read_file(os.path.join(REPO_TOP_PATH, "datasets/KernelBench/level2/1_Conv2D_ReLU_BiasAdd.py"))
    # The same for display purpose
    EXAMPLE_NEW_ARCH_SRC = read_file(os.path.join(REPO_TOP_PATH, "datasets/KernelBench/level2/1_Conv2D_ReLU_BiasAdd.py"))
    prompt = generate_reviser_prompt(
        task_params={
            "arc_src": EXAMPLE_ARCH_SRC,
            "gpu_name": "NVIDIA A100",
            "gpu_architecture": "Ampere",
            "dtype_str": "float16",
        },
        custom_triton_kernels=EXAMPLE_NEW_ARCH_SRC,
        run_info=KernelExecResult(compiled=True, correctness=True, runtime=1.0, runtime_stats={"fast_p": 1.0}),
        experience_guidance_path=None,
    )
    print(prompt)