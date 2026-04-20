import re
from agentprompt.prompt_modules import generate_experience_guidance_prompt
from agentprompt.prompt_modules import generate_hardware_information_prompt
from agentprompt.benchmarks.KB_prompt import KB_TRITON_PROMPT
from agentprompt.benchmarks.FIT_prompt import FIT_TRITON_PROMPT
from agentprompt.benchmarks.TBG_prompt import TBG_TRITON_PROMPT
from src.utils import read_file
import os

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

def _extract_format_keys(template: str):
    """Extract format keys from a template string."""
    return set(re.findall(r'\{(\w+)\}', template))

task_to_prompt = {
    "KB": KB_TRITON_PROMPT,
    "FIT": FIT_TRITON_PROMPT,
    "SYN": KB_TRITON_PROMPT,
    "MLSYS": FIT_TRITON_PROMPT,
    "TBG": TBG_TRITON_PROMPT,
}

# Generate the kernel

# Parts
# - Problem statement
# - Hardware information
# - Example Formats
# - Kernels from previous iterations

PROBLEM_STATEMENT = """## Problem Statement

You write custom kernels to replace the pytorch operators in the given architecture to get speedups.

You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You may also reorder mathematically equivalent operations to enable better fusion or memory access patterns (for example, rearranging the order of elementwise ops, or folding normalization parameters into preceding linear layers). You are only limited by your imagination.

"""

EXAMPLE_FORMATS = """## Example Formats

Here's an example to show you the syntax of inline embedding custom Triton kernels in torch: The example given architecture is:

```
{example_arch_src}
```

The example new arch with custom Triton kernels looks like this:

```
{example_new_arch_src}
```

"""

EXPERIENCE_GUIDANCE = """## Experience Guidance

Here is some experience guidance that you should keep in mind:

{experience_guidance}

"""

POOL_PROMPT = """## Kernel Pool

Here are some community-developed kernels and their runtime metrics. These represent strong baseline implementations for the given task.

Your goal is to generate a kernel that OUTPERFORMS all kernels in the pool.

You are allowed to reuse, adapt, or directly build upon any kernel in the pool. You may combine ideas, modify implementations, or even start from the best-performing kernel and improve it further.

Objective:
- Minimize runtime as much as possible.
- Your kernel should aim to beat the fastest kernel in the pool.

<Kernels and their Runtime Metrics>
{pool_kernels_and_metrics}
</Kernels and their Runtime Metrics>

Now generate a kernel that can potentially outperform the best existing kernel and achieves the lowest possible runtime.
"""

def generate_proposer_prompt(experience_guidance_path: str=None, pool_prompt: str=None, task: str="KB", task_params: dict=None, knowledge_1_threshold: int=3):
    prompt = PROBLEM_STATEMENT

    if task_params.get("example_arch_src", None) is not None and task_params.get("example_new_arch_src", None) is not None and task == "KB":
        prompt += EXAMPLE_FORMATS.format(example_arch_src=task_params.get("example_arch_src"), example_new_arch_src=task_params.get("example_new_arch_src"))

    prompt += generate_experience_guidance_prompt(experience_guidance_path, threshold=knowledge_1_threshold)
    
    # Extract required parameters from task prompt template
    task_template = task_to_prompt[task]
    required_keys = _extract_format_keys(task_template)
    
    # Build format dict: use task_params if provided, otherwise fall back to original parameters
    format_dict = {}
    if task_params is not None:
        format_dict.update(task_params)
    
    for key in required_keys:
        if key not in format_dict:
            raise ValueError(f"Missing required parameter: {key}")

    prompt += task_template.format(**format_dict)
    if pool_prompt is not None:
        prompt += pool_prompt
    return prompt

def generate_pool_prompt(
    pool_kernels: list,
    pool_metrics: list,
    *,
    proposal_ids: list[int] | None = None,
):
    if len(pool_kernels) == 0:
        return ""
    pool_kernels_and_metrics = "\n\n".join(
        [
            (
                f"\n### {i}-th kernel"
                + (
                    f" (proposal_id={proposal_ids[i]})"
                    if proposal_ids is not None and i < len(proposal_ids)
                    else ""
                )
                + ":\n\n```python\n"
                + f"{kernel}\n"
                + "```\n\n"
                + f"### {i}-th metrics:\n{metric}"
            )
            for i, (kernel, metric) in enumerate(zip(pool_kernels, pool_metrics))
        ]
    )
    return POOL_PROMPT.format(pool_kernels_and_metrics=pool_kernels_and_metrics)


def generate_pool_prompt_dual(
    *,
    kernel_pool: list,
    metrics_pool: list,
    kernel_pool_ids: list[int] | None = None,
    elite_kernel_pool: list | None = None,
    elite_metrics_pool: list | None = None,
    elite_pool_ids: list[int] | None = None,
):
    """
    Build a single Kernel Pool prompt that can include both a "recent/trajectory" pool
    and an "elite/best" pool. Any pool can be empty/None.
    """
    elite_kernel_pool = elite_kernel_pool or []
    elite_metrics_pool = elite_metrics_pool or []

    parts = []

    elite_part = generate_pool_prompt(
        elite_kernel_pool, elite_metrics_pool, proposal_ids=elite_pool_ids
    )
    if elite_part:
        inner = elite_part.split("<Kernels and their Runtime Metrics>")[1].split(
            "</Kernels and their Runtime Metrics>"
        )[0]
        parts.append(("## Context type: elite\n\n" + inner.strip()).strip())

    recent_part = generate_pool_prompt(kernel_pool, metrics_pool, proposal_ids=kernel_pool_ids)
    if recent_part:
        # Strip the outer POOL_PROMPT wrapper so we can merge into one wrapper.
        inner = recent_part.split("<Kernels and their Runtime Metrics>")[1].split(
            "</Kernels and their Runtime Metrics>"
        )[0]
        parts.append(("## Context type: recent\n\n" + inner.strip()).strip())

    if len(parts) == 0:
        return ""

    merged = "\n\n".join(parts)
    return POOL_PROMPT.format(pool_kernels_and_metrics=merged)

if __name__ == "__main__":
    print("Generating proposer prompt for KB...")
    print("-" * 100)
    EXAMPLE_ARCH_SRC = read_file(os.path.join(REPO_TOP_PATH, "agentprompt/examples/model_ex_add.py"))
    EXAMPLE_NEW_ARCH_SRC = read_file(os.path.join(REPO_TOP_PATH, "agentprompt/examples/model_new_ex_add_triton.py"))
    prompt = generate_proposer_prompt(
        task_params={
            "arc_src": EXAMPLE_ARCH_SRC,
            "gpu_name": "NVIDIA A100",
            "gpu_architecture": "Ampere",
            "dtype_str": "float16",
            "example_arch_src": EXAMPLE_ARCH_SRC,
            "example_new_arch_src": EXAMPLE_NEW_ARCH_SRC,
        },
        experience_guidance_path=None,
        pool_prompt=None,
        task="KB",
    )
    print(prompt)
    print("-" * 100)
    print("Generating proposer prompt for FIT...")
    EXAMPLE_DEFINITION = read_file(os.path.join(REPO_TOP_PATH, "datasets", "flashinfer-trace", "definitions", "gemm", "gemm_n128_k2048.json"))
    print("-" * 100)
    prompt = generate_proposer_prompt(
        task_params={
            "definition": EXAMPLE_DEFINITION,
            "target_gpu": "NVIDIA A100",
        },
        experience_guidance_path=None,
        pool_prompt=None,
        task="FIT",
    )
    print(prompt)
    print("-" * 100)
