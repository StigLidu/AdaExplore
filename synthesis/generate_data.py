import os
import sys
import time
import random
import argparse
import tempfile
import importlib.util
import json
import subprocess
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from agent.inference_server import create_inference_server, query_inference_server
from src.utils import read_file, extract_first_code
from src.dataset import construct_kernelbench_dataset

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

def generate_self_instruct_prompt(examples: list[str], num_examples: int) -> str:
    """
    Generate a Self-Instruct style prompt that asks the LLM to create new examples
    similar to the provided ones.
    """
    prompt = """You are given a collection of PyTorch kernel examples from a dataset. Each example is a complete Python file that defines:
1. A Model class (inheriting from nn.Module) 
2. A get_inputs() function that returns input tensors
3. A get_init_inputs() function that returns initialization parameters
4. Configuration variables (batch_size, dimensions, etc.)

Your task is to generate NEW, CREATIVE, and FUNCTIONALLY DIFFERENT examples that follow the same structure and style but implement different operations or variations.

Guidelines:
- Each example should be a complete, self-contained Python file
- The Model class should implement a different operation or combination of operations
- Vary the tensor shapes, dimensions, and parameters
- Keep the same code structure and style
- Make sure the examples are diverse - don't just copy the patterns exactly

Here are the example files:

"""
    for i, example in enumerate(examples[:num_examples], 1):
        prompt += f"=== Example {i} ===\n{example}\n\n"
    
    prompt += """\nNow generate a NEW example that follows the same structure but implements a different operation. 
Return ONLY the Python code, wrapped in ```python code blocks. Do not include any explanations or markdown formatting outside the code block."""

    return prompt

def generate_mutation_prompt(examples: list[str], num_examples: int) -> str:
    """
    Generate a prompt that asks the LLM to mutate existing examples by:
    - Changing tensor dimensions
    - Combining different operations
    - Modifying the operation sequence
    """
    prompt = """You are given PyTorch kernel examples. Your task is to create a NEW example by intelligently mutating and selecting elements from the provided examples.

Mutation strategies you can use:
1. Change tensor dimensions (batch size, channels, height, width, etc.)
2. Select operations from different examples
3. Modify the operation sequence or add/remove operations
4. Change parameter values (kernel sizes, strides, etc.)
5. Create variations with different activation functions or normalization layers

Each example file contains:
- A Model class (inheriting from nn.Module)
- A get_inputs() function 
- A get_init_inputs() function
- Configuration variables

Here are the example files to mutate:

"""
    for i, example in enumerate(examples[:num_examples], 1):
        prompt += f"=== Example {i} ===\n{example}\n\n"
    
    prompt += """\nGenerate a NEW example by mutating and selecting elements from the above examples. 
The result should be a complete Python file that:
- Follows the same structure
- Uses a creative selection of the operations shown
- Has different tensor shapes or parameters
- Is functionally distinct from all the examples above
- Maintain a similar level of complexity as the examples provided above. Avoid merging all operations into a single example.
Return ONLY the Python code, wrapped in ```python code blocks."""

    return prompt

def generate_creative_prompt(examples: list[str], num_examples: int) -> str:
    """
    Generate a prompt that asks the LLM to create creative variations while maintaining
    the core structure and ensuring the examples are useful for kernel optimization.
    """
    prompt = """You are helping to expand a dataset of PyTorch kernels for GPU optimization research. 
Each example represents a computational kernel that might need optimization.

Your task: Create a NEW example that:
1. Follows the exact code structure of the examples below
2. Implements a DIFFERENT computational pattern (different operations, different shapes, different combinations)
3. Is realistic and useful for kernel optimization research
4. Maintains code quality and proper PyTorch patterns

Code structure requirements:
- Model class inheriting from nn.Module
- __init__ method if needed for parameters
- forward method with the computation
- get_inputs() function returning input tensors
- get_init_inputs() function returning initialization parameters  
- Configuration variables at module level

Example files:

"""
    for i, example in enumerate(examples[:num_examples], 1):
        prompt += f"=== Example {i} ===\n{example}\n\n"
    
    prompt += """\nGenerate a NEW, creative example following this structure. 
Focus on creating something that would be useful for kernel optimization research - 
varied shapes, interesting operation combinations, and realistic use cases.

Return ONLY the Python code, wrapped in ```python code blocks."""

    return prompt

def load_pytorch_layers(json_path: str = None) -> dict:
    """
    Load PyTorch layers and activations from JSON file.
    Returns a dict with 'layers' and 'activations' keys.
    """
    if json_path is None:
        json_path = os.path.join(REPO_TOP_PATH, "skill_memory", "pytorch_layers_activations.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"PyTorch layers JSON file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def generate_composite_prompt(examples: list[str], num_examples: int, layers: list[dict] = None) -> str:
    """
    Generate a prompt that combines existing examples with random PyTorch layers
    to create more complex composite modules.
    
    Args:
        examples: List of example code strings
        num_examples: Number of examples to use
        layers: List of layer dictionaries (name, description, etc.)
    """
    prompt = """You are tasked with creating a NEW, COMPLEX PyTorch kernel module that combines:
1. Patterns and structures from the provided example files
2. PyTorch layers/operations from the provided layer list

Your goal is to create a more complex module that:
- Follows the same code structure as the examples (Model class, get_inputs, get_init_inputs, etc.)
- Incorporates one or more of the provided PyTorch layers in creative ways
- Creates a functionally different computation pattern with similar difficulty level as the examples above (3-6 operations are recommended)
- Maintains proper PyTorch patterns and code quality

Code structure requirements:
- Model class inheriting from nn.Module
- __init__ method if needed for parameters
- forward method with the computation
- get_inputs() function returning input tensors
- get_init_inputs() function returning initialization parameters  
- Configuration variables at module level

Example files (for structure reference):
"""
    
    for i, example in enumerate(examples[:num_examples], 1):
        prompt += f"\n=== Example {i} ===\n{example}\n"
    
    if layers:
        prompt += "\n\nAvailable PyTorch Layers/Operations to incorporate:\n"
        prompt += "(You should use one or more of these in your generated module)\n\n"
        for i, layer in enumerate(layers[:num_examples], 1):
            prompt += f"{i}. {layer['name']}\n"
            if layer.get('description'):
                prompt += f"   Description: {layer['description']}\n"
            if layer.get('section'):
                prompt += f"   Category: {layer['section']}\n"
            prompt += "\n"
    
    prompt += """\nGenerate a NEW, complex example that:
- Uses the structure from the examples above
- Incorporates one or more of the provided PyTorch layers
- Creates a more complex computation pattern (e.g., combining multiple layers, using different tensor shapes, etc.)
- Is functionally distinct from all the examples above
- Maintains similar or higher complexity level as the examples

Return ONLY the Python code, wrapped in ```python code blocks. Do not include any explanations or markdown formatting outside the code block."""

    return prompt

PROMPT_STYLES = {
    "self_instruct": generate_self_instruct_prompt,
    "mutation": generate_mutation_prompt,
    "creative": generate_creative_prompt,
    "composite": generate_composite_prompt,
}

def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize model name for use in file paths by replacing invalid characters.
    """
    # Replace characters that might cause issues in file paths
    sanitized = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    sanitized = sanitized.replace("*", "_").replace("?", "_").replace('"', "_")
    sanitized = sanitized.replace("<", "_").replace(">", "_").replace("|", "_")
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    return sanitized

def validate_generated_kernel(generated_code: str) -> tuple[bool, str]:
    """
    Validate that a generated kernel module can be executed successfully.
    Uses the same execution harness as KernelBench.
    
    Returns:
        (is_valid, error_message): tuple of validation result and error message if any
    """
    # Check for basic required components
    if "class Model" not in generated_code:
        return False, "Missing 'class Model'"
    if "def forward" not in generated_code:
        return False, "Missing 'def forward'"
    if "def get_inputs" not in generated_code:
        return False, "Missing 'def get_inputs'"
    if "def get_init_inputs" not in generated_code:
        return False, "Missing 'def get_init_inputs'"

    if not torch.cuda.is_available():
        return False, "CUDA is not available (GPU is required for validation)"
    
    # Create a temporary file to load the module
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
            tmp_file.write(generated_code)
            tmp_file_path = tmp_file.name
        
        try:
            # Dynamically import the generated module for validation
            module_name = f"generated_kernel_{random.randint(1000, 9999)}"
            spec = importlib.util.spec_from_file_location(module_name, tmp_file_path)
            if spec is None or spec.loader is None:
                return False, "Failed to create module spec"
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get required attributes
            try:
                Model = getattr(module, "Model")
                get_inputs = getattr(module, "get_inputs")
                get_init_inputs = getattr(module, "get_init_inputs")
            except AttributeError as e:
                return False, f"Missing required attribute: {e}"
            
            # Try to execute the model with test inputs
            with torch.no_grad():
                try:
                    inputs = get_inputs()
                except Exception as e:
                    return False, f"Error in get_inputs(): {e}"
                
                try:
                    init_inputs = get_init_inputs()
                except Exception as e:
                    return False, f"Error in get_init_inputs(): {e}"
                
                # Move inputs to CUDA (GPU validation only)
                inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]
                init_inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in init_inputs]
                device = torch.device("cuda")
                
                # Initialize and run the model
                try:
                    model = Model(*init_inputs)
                    model = model.to(device)
                except Exception as e:
                    return False, f"Error initializing Model: {e}"
                
                try:
                    output = model(*inputs)
                    # Check that output is a tensor
                    if not isinstance(output, torch.Tensor):
                        return False, f"Model output is not a torch.Tensor, got {type(output)}"
                    # Check that output has valid shape
                    if output.numel() == 0:
                        return False, "Model output is empty"
                except Exception as e:
                    return False, f"Error running model forward(): {e}"
            
            # If we get here, everything worked!
            return True, ""
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
                
    except Exception as e:
        return False, f"Error during validation: {e}"

def validate_generated_kernel_subprocess(
    generated_code: str, 
    timeout: int = 300,  # 5 minutes default timeout
) -> tuple[bool, str]:
    """
    Validate that a generated kernel module can be executed successfully using a subprocess.
    This isolates runtime errors and GPU issues, preventing them from crashing the main process.
    
    Returns:
        (is_valid, error_message): tuple of validation result and error message if any
    """
    # Get the path to the subprocess runner script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, 'valid_subprocess_runner.py')
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(
            f"Subprocess runner script not found at {script_path}"
        )
    
    # Prepare arguments as JSON
    args_dict = {
        'generated_code': generated_code,
    }
    args_json = json.dumps(args_dict)
    
    # Get the repo root directory
    repo_root = os.path.dirname(script_dir)
    
    # Run the subprocess
    process = subprocess.Popen(
        [sys.executable, script_path, args_json],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=repo_root
    )
    
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        stdout_str = stdout.decode('utf-8')
        stderr_str = stderr.decode('utf-8')
        
        # Parse the JSON result
        try:
            result_dict = json.loads(stdout_str)
            is_valid = result_dict.get('is_valid', False)
            error_message = result_dict.get('error_message', '')
            return is_valid, error_message
        except json.JSONDecodeError:
            # If JSON parsing fails, return error with stderr
            error_msg = f"Failed to parse subprocess output. stdout: {stdout_str[:500]}, stderr: {stderr_str[:500]}"
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return False, f"Validation timeout after {timeout} seconds"
    except Exception as e:
        return False, f"Subprocess error: {e}"

def load_kernel_files(levels: list[int], num_examples_per_request: int = 3, max_idx: int = 50) -> dict[int, list[str]]:
    """
    Load kernel files from specified levels.
    Returns a dict mapping level -> list of file contents.
    """
    level_files = {}
    for level in levels:
        dataset = construct_kernelbench_dataset(level)
        files = []
        for file_path in dataset:
            idx = file_path.split("/")[-1].split("_")[0]
            if int(idx) > max_idx:
                continue
            content = read_file(file_path)
            if content:
                files.append(content)
        level_files[level] = files
        print(f"Loaded {len(files)} files from level {level}")
    return level_files

def generate_data(
    inference_server,
    model_name: str,
    prompt_style: str,
    level_files: dict[int, list[str]],
    num_generations: int,
    num_examples_per_request: int,
    output_dir: str,
    max_completion_tokens: int = 16384,
    temperature: float = 1.0,
    validate_execution: bool = True,
    layers_data: dict = None,
    use_subprocess_validation: bool = True,
):
    """
    Generate new kernel examples using LLM.
    
    Args:
        layers_data: Dictionary with 'layers' and 'activations' keys for composite prompt style
        use_subprocess_validation: If True, use subprocess for validation to isolate GPU errors
    """
    prompt_fn = PROMPT_STYLES[prompt_style]
    os.makedirs(output_dir, exist_ok=True)

    # Load layers data if using composite prompt style
    if prompt_style == "composite" and layers_data is None:
        try:
            layers_data = load_pytorch_layers()
            print(f"Loaded {len(layers_data.get('layers', []))} layers and {len(layers_data.get('activations', []))} activations")
        except Exception as e:
            print(f"Warning: Failed to load layers data: {e}")
            print("Falling back to regular prompt style")
            layers_data = None

    # Track which level we're generating for (alternate or random)
    all_levels = list(level_files.keys())
    
    existing_kernel_num = len(list(Path(output_dir).glob("*.py")))
    print(f"Existing kernels: {existing_kernel_num}")
    generated_count = existing_kernel_num
    max_failures = (num_generations - existing_kernel_num) * 2
    failed_count = 0
    attempt_count = 0  # Track total attempts (including failed ones)
    
    with tqdm(total=num_generations - existing_kernel_num, desc="Generating kernels") as pbar:
        while generated_count < num_generations:
            attempt_count += 1
            # Randomly select a level
            level = random.choice(all_levels)
            files = level_files[level]
            
            # Randomly select examples from this level
            if len(files) < num_examples_per_request:
                selected_examples = files
            else:
                selected_examples = random.sample(files, num_examples_per_request)
            
            # Generate prompt
            if prompt_style == "composite" and layers_data:
                # For composite style, also select random layers
                all_layers = layers_data.get('layers', []) + layers_data.get('activations', [])
                if len(all_layers) < num_examples_per_request:
                    selected_layers = all_layers
                else:
                    selected_layers = random.sample(all_layers, num_examples_per_request)
                prompt = prompt_fn(selected_examples, num_examples_per_request, selected_layers)
            else:
                prompt = prompt_fn(selected_examples, num_examples_per_request)
            
            # Save prompt to file for debugging
            # prompt_filename = f"prompt_{level}_{attempt_count}.txt"
            # prompt_path = os.path.join(output_dir, prompt_filename)
            # with open(prompt_path, "w") as f:
            #     f.write(prompt)
            
            try:
                # Query LLM
                response = query_inference_server(
                    inference_server,
                    model_name=model_name,
                    prompt=prompt,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                )
                
                # Extract code
                generated_code = extract_first_code(response, ["python"])
                
                # Validate the generated code
                is_valid = False
                validation_error = ""
                
                if validate_execution:
                    # Full execution validation
                    if use_subprocess_validation:
                        is_valid, validation_error = validate_generated_kernel_subprocess(
                            generated_code, 
                        )
                    else:
                        is_valid, validation_error = validate_generated_kernel(
                            generated_code,
                        )
                    if not is_valid:
                        print(f"Validation failed: {validation_error}")
                else:
                    # Basic syntax validation only
                    if "class Model" in generated_code and "def forward" in generated_code:
                        is_valid = True
                
                if is_valid:
                    # Save the generated file
                    output_filename = f"{prompt_style}_{level}_{generated_count + 1}.py"
                    output_path = os.path.join(output_dir, output_filename)

                    with open(output_path, "w") as f:
                        f.write(generated_code)

                    # Also save the response (before code extraction) for debugging
                    response_filename = f"response_{prompt_style}_{level}_{generated_count + 1}.txt"
                    response_path = os.path.join(output_dir, response_filename)
                    with open(response_path, "w") as f:
                        f.write(response)
                    
                    # Save the prompt to file for debugging
                    prompt_filename = f"prompt_{prompt_style}_{level}_{generated_count + 1}.txt"
                    prompt_path = os.path.join(output_dir, prompt_filename)
                    with open(prompt_path, "w") as f:
                        f.write(prompt)

                    generated_count += 1
                    pbar.update(1)
                    validation_msg = " (validated)" if validate_execution else ""
                    print(f"Generated {output_filename}{validation_msg} ({generated_count}/{num_generations})")
                else:
                    reason = validation_error if validate_execution else "basic structure check failed"
                    print(f"Warning: Generated code validation failed ({reason}), skipping...")
                    failed_count += 1
                    if failed_count > max_failures:  # Give up after too many failures
                        print(f"Too many failures, stopping generation")
                        break

            except Exception as e:
                print(f"Error during generation: {e}")
                failed_count += 1
                if failed_count > max_failures:
                    print(f"Too many failures, stopping generation")
                    break
                continue
    
    print(f"\nGeneration complete!")
    print(f"Successfully generated: {generated_count - existing_kernel_num}/{num_generations}")
    print(f"Failed: {failed_count}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate new kernel data using LLM")
    parser.add_argument("--server_type", type=str, default="azure", choices=["azure", "openai", "claude"])
    parser.add_argument("--model_name", type=str, default="gpt-5-mini", help="Model name to use")
    parser.add_argument("--prompt_style", type=str, default="composite", 
                        choices=list(PROMPT_STYLES.keys()),
                        help="Prompt style to use for generation")
    parser.add_argument("--num_generations", type=int, default=10, 
                        help="Number of new examples to generate")
    parser.add_argument("--num_examples_per_request", type=int, default=3,
                        help="Number of example files to include in each prompt")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for generated files")
    parser.add_argument("--max_completion_tokens", type=int, default=16384,
                        help="Maximum completion tokens")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for generation")
    parser.add_argument("--skip_validation", action="store_true", default=False,
                        help="Skip execution validation (only check basic structure)")
    parser.add_argument("--no_subprocess_validation", action="store_true", default=False,
                        help="Disable subprocess validation (use direct validation instead, may cause GPU errors to crash main process)")
    parser.add_argument("--input_levels", nargs='+', type=int, default=[1],
                        help="Levels to load kernel files from")
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        sanitized_model_name = sanitize_model_name(args.model_name)
        args.output_dir = os.path.join(
            REPO_TOP_PATH, 
            "outputs",
            "data_generation",
            f"generated_data_{args.prompt_style}_{sanitized_model_name}_{start_time}"
        )

    print(f"Loading kernel files from levels: {args.input_levels}")
    level_files = load_kernel_files(levels=args.input_levels, num_examples_per_request=args.num_examples_per_request)
    
    print(f"Creating inference server ({args.server_type})...")
    inference_server = create_inference_server(args.server_type)
    print("Inference server created successfully!")
    
    print(f"\nGenerating {args.num_generations} new examples using '{args.prompt_style}' prompt style...")
    print(f"Output directory: {args.output_dir}\n")
    
    validate_execution = not args.skip_validation
    
    use_subprocess_validation = not args.no_subprocess_validation
    
    if validate_execution:
        print(f"Execution validation: enabled")
        if use_subprocess_validation:
            print(f"Using subprocess validation (isolates GPU errors)")
        else:
            print(f"Using direct validation (GPU errors may crash main process)")
        if torch.cuda.is_available():
            print(f"CUDA available: Using GPU for validation")
        else:
            raise SystemExit("CUDA is required for validation but not available. Use --skip_validation to run without validation.")
    else:
        print(f"Execution validation: disabled (using basic structure check only)")
    
    # Load layers data if using composite style
    layers_data = None
    if args.prompt_style == "composite":
        try:
            layers_data = load_pytorch_layers()
            print(f"Loaded {len(layers_data.get('layers', []))} layers and {len(layers_data.get('activations', []))} activations for composite generation")
        except Exception as e:
            print(f"Warning: Failed to load layers data: {e}")
            print("Composite prompt style requires layers data. Please ensure skill_memory/pytorch_layers_activations.json exists.")
    
    generate_data(
        inference_server=inference_server,
        model_name=args.model_name,
        prompt_style=args.prompt_style,
        level_files=level_files,
        num_generations=args.num_generations,
        num_examples_per_request=args.num_examples_per_request,
        output_dir=args.output_dir,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        validate_execution=validate_execution,
        layers_data=layers_data,
        use_subprocess_validation=use_subprocess_validation,
    )