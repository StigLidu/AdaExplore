"""
Subprocess runner for validate_generated_kernel
This script is executed in a separate process to isolate runtime errors and GPU issues
"""
import sys
import os
import json
import torch
import tempfile
import importlib.util
import random

# Get the parent directory (repo root) so we can import from synthesis
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == 'synthesis':
    repo_root = os.path.dirname(script_dir)
else:
    repo_root = script_dir

# Add the parent directory to the path so we can import from synthesis
sys.path.insert(0, repo_root)

# Parse arguments from JSON
args = json.loads(sys.argv[1])
generated_code = args['generated_code']

# Always require CUDA for execution validation
if not torch.cuda.is_available():
    result_dict = {
        'is_valid': False,
        'error_message': "CUDA is not available (GPU is required for validation)"
    }
    print(json.dumps(result_dict))
    sys.exit(0)

# Check for basic required components
if "class Model" not in generated_code:
    result_dict = {
        'is_valid': False,
        'error_message': "Missing 'class Model'"
    }
    print(json.dumps(result_dict))
    sys.exit(0)

if "def forward" not in generated_code:
    result_dict = {
        'is_valid': False,
        'error_message': "Missing 'def forward'"
    }
    print(json.dumps(result_dict))
    sys.exit(0)

if "def get_inputs" not in generated_code:
    result_dict = {
        'is_valid': False,
        'error_message': "Missing 'def get_inputs'"
    }
    print(json.dumps(result_dict))
    sys.exit(0)

if "def get_init_inputs" not in generated_code:
    result_dict = {
        'is_valid': False,
        'error_message': "Missing 'def get_init_inputs'"
    }
    print(json.dumps(result_dict))
    sys.exit(0)

try:
    # Create a temporary file to load the module
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(generated_code)
        tmp_file_path = tmp_file.name
    
    try:
        # Dynamically import the generated module for validation
        module_name = f"generated_kernel_{random.randint(1000, 9999)}"
        spec = importlib.util.spec_from_file_location(module_name, tmp_file_path)
        if spec is None or spec.loader is None:
            result_dict = {
                'is_valid': False,
                'error_message': "Failed to create module spec"
            }
            print(json.dumps(result_dict))
            sys.exit(0)
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get required attributes
        try:
            Model = getattr(module, "Model")
            get_inputs = getattr(module, "get_inputs")
            get_init_inputs = getattr(module, "get_init_inputs")
        except AttributeError as e:
            result_dict = {
                'is_valid': False,
                'error_message': f"Missing required attribute: {e}"
            }
            print(json.dumps(result_dict))
            sys.exit(0)
        
        # Try to execute the model with test inputs
        with torch.no_grad():
            try:
                inputs = get_inputs()
            except Exception as e:
                result_dict = {
                    'is_valid': False,
                    'error_message': f"Error in get_inputs(): {e}"
                }
                print(json.dumps(result_dict))
                sys.exit(0)
            
            try:
                init_inputs = get_init_inputs()
            except Exception as e:
                result_dict = {
                    'is_valid': False,
                    'error_message': f"Error in get_init_inputs(): {e}"
                }
                print(json.dumps(result_dict))
                sys.exit(0)
            
            # Move inputs to CUDA (GPU validation only)
            inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]
            init_inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in init_inputs]
            device = torch.device("cuda")
            
            # Initialize and run the model
            try:
                model = Model(*init_inputs)
                model = model.to(device)
            except Exception as e:
                result_dict = {
                    'is_valid': False,
                    'error_message': f"Error initializing Model: {e}"
                }
                print(json.dumps(result_dict))
                sys.exit(0)
            
            try:
                output = model(*inputs)
                # Check that output is a tensor
                if not isinstance(output, torch.Tensor):
                    result_dict = {
                        'is_valid': False,
                        'error_message': f"Model output is not a torch.Tensor, got {type(output)}"
                    }
                    print(json.dumps(result_dict))
                    sys.exit(0)
                # Check that output has valid shape
                if output.numel() == 0:
                    result_dict = {
                        'is_valid': False,
                        'error_message': "Model output is empty"
                    }
                    print(json.dumps(result_dict))
                    sys.exit(0)
            except Exception as e:
                result_dict = {
                    'is_valid': False,
                    'error_message': f"Error running model forward(): {e}"
                }
                print(json.dumps(result_dict))
                sys.exit(0)
        
        # If we get here, everything worked!
        result_dict = {
            'is_valid': True,
            'error_message': ""
        }
        print(json.dumps(result_dict))
        sys.exit(0)
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_file_path)
        except:
            pass
            
except Exception as e:
    import traceback
    result_dict = {
        'is_valid': False,
        'error_message': f"Error during validation: {e}",
        'traceback': traceback.format_exc(),
        'error_type': type(e).__name__
    }
    print(json.dumps(result_dict))
    sys.exit(1)

