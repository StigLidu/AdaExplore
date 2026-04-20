# Generate the kernel

# Parts
# - Problem statement
# - Hardware information
# - Example Formats
# - Kernels from previous iterations

from src.utils import read_file
import os

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

from src.eval import KernelExecResult, eval_kernel_against_ref
from agent.inference_server import create_inference_server, query_inference_server
from agent.utils import extract_edits, str_replace
import re
from agentprompt.prompt_modules import generate_experience_guidance_prompt
from agentprompt.prompt_modules import generate_hardware_information_prompt

def _extract_format_keys(template: str):
    """Extract format keys from a template string."""
    return set(re.findall(r'\{(\w+)\}', template))

PROBLEM_STATEMENT = """## Problem Statement

You tune the custom Triton kernels in the given architecture to get better performance. The architecture is the reference architecture, and the custom kernels are the previous kernels you have generated. Beyond low-level kernel tuning, you may also restructure the computation order when it is mathematically equivalent and can unlock better fusion or memory access patterns.

"""

TASK_INSTRUCTION = """## Task Instruction

You are given the following architecture:

```python
{arc_src}
```

The input shapes can be found in the input of the architecture, and the dtype is {dtype_str}.

The tuning metrics contain the following information:

* **Compiled**: whether the kernel is compiled successfully
* **Error Message**: the compilation or runtime error encountered by the kernel (if any)
* **Correctness**: whether the kernel is correct
* **Runtime**: the runtime of the kernel
* **Fast_p**: compared with the standard PyTorch implementation, how much speedup the customized kernel achieves, calculated as *standard time / custom time*.

### Test Conditions

* **Correctness Test:**
  First, verify the correctness of the custom kernels by running each kernel with the specified input shapes and data types.

* **Warm-up Phase:**
  Warm up the kernel by running it three times with the same input shapes and data types.
  The runtime during the warm-up phase is **not** included in the final runtime, so you may include auto-tuning code as part of this phase.

* **Performance Test:**
  Finally, test the performance of the custom kernels by running each kernel 100 times with the same input shapes and data types.
  The runtime from these test runs **is included** in the final performance measurement.

### Goal

- Perform small, localized updates to code in the last version of the custom kernels with the str_replace command to correct the correctness errors or improve the performance of the custom kernels. Keep the overall model interface unchanged, but you may reorder or restructure internal operations when mathematically equivalent for better performance.
When making edits:
   - Ensure the edit results in idiomatic, correct code
   - Do not leave the code in a broken state

CRITICAL REQUIREMENTS FOR USING THIS TOOL:

1. EXACT MATCHING: The `old_str` parameter must match EXACTLY one or more consecutive lines from the file, including all whitespace and indentation. 
- You should ensure the `old_str` matches exactly with the file content, otherwise the str_replace tool will fail.

2. UNIQUENESS: The `old_str` must uniquely identify a single instance in the file:
   - Include sufficient context before and after the change point (3-5 lines recommended)
   - If not unique, the replacement will not be performed

3. REPLACEMENT: The `new_str` parameter should contain the edited lines that replace the `old_str`. Both strings must be different.

Remember: You should prefer to send all edits in a single message with multiple calls rather than multiple messages with a single call each.

#### **Output Format**:

You should output all the edits in a single message with multiple call. Each call should be a single edit as follows with id `1` to `n`. 
- For each edit, you should provide the reasoning for the edit in the <reasoning_i> block, followed by the old code block in the <old_str_i> block, followed by the new code block in the <new_str_i> block.
- You should ensure the `old_str_i` matches exactly with the file content, otherwise the str_replace tool will fail.

Example output format:

<reasoning_1>
// reasoning for the edit 1
...
</reasoning_1>
<old_str_1>
// old code block (must match exactly)
...
</old_str_1>
<new_str_1>
// new code block
...
</new_str_1>

...

<reasoning_n>
// reasoning for the edit n
...
</reasoning_n>
<old_str_n>
// old code block (must match exactly)
...
</old_str_n>
<new_str_n>
// new code block
...
</new_str_n>

#### **Previous Kernels and Metrics:**

Previously, you have generated the following custom kernels and got the following runtime metrics:

<Previous Kernels and Metrics>
{previous_kernels_and_metrics}
</Previous Kernels and Metrics>

#### **Guidance from Reviser Agent:**

- A reviser agent has provided a guidance to you on how to correct or improve the performance of your lastgenerated custom kernels, please tune the custom kernels based on the guidance in the unified diff format. Reminder to keep the name of `ModelNew` unchanged.

<Tuning Guidance>
{tuning_guidance}
</Tuning Guidance>

"""

HARDWARE_INFORMATION = """## Hardware Information

Here is some information about the underlying hardware that you should keep in mind:

- The GPU that will run the kernel is NVIDIA {gpu_name}, {gpu_architecture} architecture.

"""

EXPERIENCE_GUIDANCE = """## Experience Guidance

Here is some experience guidance that you should keep in mind:

{experience_guidance}

"""

def _is_correct_metric(metric) -> bool:
    """Check if a metric indicates correctness. Handles both KernelExecResult objects and string representations."""
    if isinstance(metric, KernelExecResult):
        return metric.correctness
    elif isinstance(metric, str):
        # Check string representation for correctness
        return "correctness=True" in metric or '"correctness": true' in metric.lower()
    return False

def generate_tuner_prompt(
    previous_kernels: list[str] = None,
    previous_metrics: list[str] = None,
    tuning_guidance: str = None,
    experience_guidance_path: str = None,
    filter_wrong_attempts: bool = False,
    task_params: dict = None,
    knowledge_1_threshold: int = 3,
):
    # Filter out wrong attempts if requested
    if filter_wrong_attempts:
        filtered_pairs = [
            (kernel, metric) 
            for kernel, metric in zip(previous_kernels, previous_metrics)
            if _is_correct_metric(metric)
        ]
        if filtered_pairs:
            previous_kernels, previous_metrics = zip(*filtered_pairs)
            previous_kernels, previous_metrics = list(previous_kernels), list(previous_metrics)
        else:
            previous_kernels, previous_metrics = [], []
    
    previous_kernels_and_metrics_str = "\n".join(
        [f"\n### {i}-th attempt: \n\n```python\n{kernel}\n```\n\n### {i}-th Runtime Metrics:\n{metric}" \
            for i, (kernel, metric) in enumerate(zip(previous_kernels, previous_metrics))])
    
    # Extract required parameters from task prompt template
    required_keys = _extract_format_keys(TASK_INSTRUCTION)
    
    # Build format dict: use task_params if provided, otherwise fall back to original parameters
    format_dict = {}
    if task_params is not None:
        format_dict.update(task_params)
    
    # Fall back to original parameters for missing keys
    for key in required_keys:
        if key not in format_dict:
            if key == 'previous_kernels_and_metrics':
                format_dict[key] = previous_kernels_and_metrics_str
            elif key == 'tuning_guidance':
                format_dict[key] = tuning_guidance
            else:
                raise ValueError(f"Missing required parameter: {key}")
    
    prompt = PROBLEM_STATEMENT
    prompt += generate_experience_guidance_prompt(experience_guidance_path, threshold=knowledge_1_threshold)
    prompt += generate_hardware_information_prompt(task_params.get('gpu_name'), task_params.get('gpu_architecture'))
    prompt += TASK_INSTRUCTION.format(**format_dict)
    return prompt

if __name__ == "__main__":
    EXAMPLE_ARCH_SRC = read_file(os.path.join(REPO_TOP_PATH, "agentprompt/examples/model_ex_add.py"))
    EXAMPLE_NEW_ARCH_SRC = read_file(os.path.join(REPO_TOP_PATH, "agentprompt/examples/model_new_ex_add_triton.py"))
    prompt = generate_tuner_prompt(
        task_params={
            "arc_src": EXAMPLE_ARCH_SRC,
            "gpu_name": "NVIDIA A100",
            "gpu_architecture": "Ampere",
            "dtype_str": "float16",
        },
        previous_kernels=[EXAMPLE_NEW_ARCH_SRC],
        previous_metrics=[KernelExecResult(compiled=True, correctness=True, runtime=1.0, runtime_stats={"fast_p": 1.0})],
        tuning_guidance=None,
        experience_guidance_path=None,
    )
    print(prompt)
    print("-" * 100)
    # call llm to get the edits
    inference_server = create_inference_server(
        server_type="azure",
    )
    output = query_inference_server(inference_server, model_name="gpt-5", prompt=prompt, max_completion_tokens=1000)
    print(output)
    print("-" * 100)
    edits = extract_edits(output)
    print(edits)
    print("-" * 100)
    # apply the edits to the example new arch src
    for old_str, new_str in edits:
        EXAMPLE_NEW_ARCH_SRC = str_replace(EXAMPLE_NEW_ARCH_SRC, old_str, new_str)
    print(EXAMPLE_NEW_ARCH_SRC)
    print("-" * 100)
    # evaluate the tuned kernel
    metrics = eval_kernel_against_ref(
        EXAMPLE_ARCH_SRC,
        EXAMPLE_NEW_ARCH_SRC,
        verbose=False,
        measure_performance=True,
        num_correct_trials=5,
        num_perf_trials=100,
        backend="triton",
        dtype_str="fp16",
    )
    print(metrics)