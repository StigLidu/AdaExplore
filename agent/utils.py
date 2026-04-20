import re
import os
import sys
import argparse
import yaml
from src.utils import read_file
from src.dataset import (
    construct_kernelbench_dataset, 
    construct_synthesized_data_dataset,
    load_flashinfer_trace_definition,
    get_flashinfer_style_dataset_root,
    construct_tritonbench_g_dataset,
    get_tritonbench_g_dataset_root,
    resolve_tritonbench_g_problem_path,
)
from src.eval import KernelExecResult
import ast
import random

def _find_occurrences(pattern: str, file_content: str) -> list:
    return [
        (
            file_content.count('\n', 0, match.start()) + 1,
            match.group(),
            match.start(),
        )
        for match in re.finditer(pattern, file_content)
    ]


def _build_trailing_ws_pattern(text: str) -> str:
    """Build a regex that matches *text* but tolerates trailing whitespace on each line."""
    lines = text.split('\n')
    return r'\n'.join(re.escape(line.rstrip()) + r'[ \t]*' for line in lines)


def str_replace(
    file_content: str,
    old_str: str,
    new_str: str | None,
    encoding: str = 'utf-8',
) -> str:
    """
    Implement the str_replace command, which replaces old_str with new_str in the file content.

    Args:
        file_content: The original file content
        old_str: String to replace
        new_str: Replacement string
        encoding: The encoding to use (auto-detected by decorator)
    """
    new_str = new_str or ''

    # 1) Exact match
    pattern = re.escape(old_str)
    occurrences = _find_occurrences(pattern, file_content)

    # 2) Fallback: tolerate trailing whitespace differences per line
    if not occurrences:
        occurrences = _find_occurrences(_build_trailing_ws_pattern(old_str), file_content)

    # 3) Fallback: strip leading/trailing blank lines, then tolerate trailing ws
    if not occurrences:
        old_stripped = old_str.strip('\n')
        if old_stripped != old_str:
            occurrences = _find_occurrences(
                _build_trailing_ws_pattern(old_stripped), file_content
            )

    if not occurrences:
        print(f"[Warning] No replacement was performed, old_str\n ```\n{old_str}\n```\ndid not appear verbatim in the file.")
        return file_content
    if len(occurrences) > 1:
        line_numbers = sorted(set(line for line, _, _ in occurrences))
        print(f"[Warning] No replacement was performed. Multiple occurrences of old_str\n ```\n{old_str}\n```\nin lines {line_numbers}. Please ensure it is unique.")
        return file_content

    replacement_line, matched_text, idx = occurrences[0]

    new_file_content = (
        file_content[:idx] + new_str + file_content[idx + len(matched_text) :]
    )

    return new_file_content


def extract_edits(output: str):
    edits = []
    pattern = r'<old_str_(\d+)>(.*?)</old_str_\1>\s*<new_str_\1>(.*?)</new_str_\1>'
    for match in re.finditer(pattern, output, re.DOTALL):
        old_str = match.group(2).strip('\n')
        new_str = match.group(3).strip('\n')
        edits.append((old_str, new_str))
    return edits

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

EXAMPLE_ARCH_SRC = read_file(
    os.path.join(REPO_TOP_PATH, "agentprompt/examples/model_ex_add.py")
)
EXAMPLE_NEW_ARCH_SRC = read_file(
    os.path.join(REPO_TOP_PATH, "agentprompt/examples/model_new_ex_add_triton.py")
)

def calculate_score(metric: KernelExecResult):
    if metric is None:
        return (0, 0, 0)
    if not metric.compiled:
        return (0, 0, 0)
    if not metric.correctness:
        return (1, 0, 0)
    else:
        fast_p = metric.runtime_stats.get("fast_p", 0) if metric.runtime_stats else 0
        return (1, 1, fast_p)
    
def dummy_metrics():
    Compiled_flag = random.random() < 0.5
    Correctness_flag = random.random() < 0.5 if Compiled_flag else False
    Speedup = random.random() * 2 if Compiled_flag and Correctness_flag else 0.0
    return KernelExecResult(
        compiled=Compiled_flag,
        correctness=Correctness_flag,
        runtime_stats={"fast_p": Speedup}
    )

def load_test_source(test_source: str, level, problem_id, suffix: str = ""):
    """
    Load reference architecture source code.
    KB/SYN: level=int, problem_id=int (1-indexed)
    FIT: level=op_type (str), problem_id=problem_name (str)
    TBG: level is ignored, problem_id can be 1-indexed id or filename
    suffix: dataset folder suffix, e.g. "_v1" for KernelBench_v1
    """
    if test_source in ("KB", "SYN"):
        dataset_fn = construct_kernelbench_dataset if test_source == "KB" else construct_synthesized_data_dataset
        dataset = dataset_fn(int(level), suffix=suffix) if test_source == "KB" else dataset_fn(int(level))
        ref_arch_path = dataset[int(problem_id) - 1]
        return os.path.basename(ref_arch_path), read_file(ref_arch_path)
    elif test_source in ("FIT", "MLSYS"):
        dataset_root = get_flashinfer_style_dataset_root(test_source)
        definition = load_flashinfer_trace_definition(str(level), str(problem_id), dataset_root=dataset_root)
        return str(problem_id), definition
    elif test_source == "TBG":
        ref_arch_path = resolve_tritonbench_g_problem_path(
            str(problem_id),
            dataset_root=get_tritonbench_g_dataset_root(),
        )
        return os.path.basename(ref_arch_path), read_file(ref_arch_path)
    else:
        raise ValueError(f"Invalid test source: {test_source}")

def compare_score(score1: tuple, score2: tuple|None=None):
    # < 1: score1 is worse than score2, >= 1: score1 is better than score2
    # < 0: much worse
    if score2 is None:
        return float('inf')
    if score2[2] == 0:
        if score1[2] == 0:
            return 1
        else:
            return float('inf')
    else:
        return score1[2] / score2[2]

def copy_step_files(src_path: str, dst_path: str):
    """
    Copy all non-best log files from source to destination directory.
    Skips files containing 'best' in the filename (e.g. global_best_kernel, global_best_metrics).
    Works for all agent types (MCTS, IRS, PS, etc.) without hardcoding file patterns.

    Args:
        src_path: Source log directory
        dst_path: Destination log directory
    """
    import shutil
    os.makedirs(dst_path, exist_ok=True)
    for fname in os.listdir(src_path):
        if "best" in fname:
            continue
        src_file = os.path.join(src_path, fname)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, os.path.join(dst_path, fname))


def read_metrics(metrics_path: str, full_metrics: bool = False):
    """
    Read metrics from a file. Supports both TXT and JSON formats.
    
    Args:
        metrics_path: Path to the metrics file (.txt or .json)
        full_metrics: If True, return full KernelExecResult; otherwise return (correctness, fast_p)
    
    Returns:
        If full_metrics: KernelExecResult object
        Otherwise: tuple (correctness: bool, fast_p: float)
    """
    with open(metrics_path, "r") as f:
        content = f.read()
    
    # Detect format by file extension or content
    is_json = metrics_path.endswith(".json") or content.strip().startswith("{")
    
    if is_json:
        import json
        data = json.loads(content)
        if full_metrics:
            return KernelExecResult(
                compiled=data.get("compiled", False),
                correctness=data.get("correctness", False),
                metadata=data.get("metadata", {}),
                runtime=data.get("runtime", -1.0),
                runtime_stats=data.get("runtime_stats", {})
            )
        else:
            compiled = data.get("compiled", False)
            correctness = data.get("correctness", False)
            if compiled and correctness:
                fast_p = data.get("runtime_stats", {}).get("fast_p", 0.0)
                return (True, fast_p)
            else:
                return (False, 0)
    else:
        # TXT format (legacy): "compiled=True correctness=True metadata={} runtime=-1.0 runtime_stats={...}"
        compiled = content.split("compiled=")[1].split(" ")[0].rstrip(",)") == "True"
        correctness = content.split("correctness=")[1].split(" ")[0].rstrip(",)") == "True"
        runtime_stats = {}
        if compiled and correctness:
            runtime_stats_str = content.split("runtime_stats=")[1].split("\n")[0].rstrip(")")
            runtime_stats = ast.literal_eval(runtime_stats_str)
        if full_metrics:
            return KernelExecResult(
                compiled=compiled,
                correctness=correctness,
                runtime_stats=runtime_stats,
            )
        else:
            if compiled and correctness:
                return (True, runtime_stats.get("fast_p", 0.0))
            else:
                return (False, 0)

DATASET_RANGES = {
    "KB": {
        "1": range(1, 101),
        "2": range(1, 101),
        "3": range(1, 51),
    },
    "SYN": {
        "1": range(1, 101),
        "2": range(1, 101),
        "3": range(1, 101),
        "4": range(1, 501),
    },
}

def load_tasks_from_test_list(test_list_path: str, test_source: str = "KB") -> list[dict]:
    """
    Load tasks from a test list file.
    
    Supported formats:
        - "2" - level 2, all problems (1-100)
        - "2 5" - level 2, problem 5
        - "2 1-50" - level 2, problems 1-50
        - "2 1-50,77-78" - level 2, problems 1-50 and 77-78
        - "2 1,3,5,7-10" - level 2, problems 1, 3, 5, and 7-10
    
    Args:
        test_list_path: Path to the test list file
        
    Returns:
        List of task dicts with 'level' and 'problem_id' keys
    """
    from src.dataset import construct_flashinfer_trace_dataset
    
    with open(test_list_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    
    tasks = []
    for line in lines:
        parts = line.split(" ", 1)
        level = parts[0]
        
        if test_source in ("FIT", "MLSYS"):
            if len(parts) == 1:
                problems = construct_flashinfer_trace_dataset(
                    level,
                    dataset_root=get_flashinfer_style_dataset_root(test_source),
                )
            else:
                problems = [p.strip() for p in parts[1].split(",") if p.strip()]
            tasks.extend({'level': level, 'problem_id': str(p)} for p in problems)
        elif test_source == "TBG":
            dataset = construct_tritonbench_g_dataset(dataset_root=get_tritonbench_g_dataset_root())
            all_problem_names = [os.path.basename(p)[:-3] for p in dataset]  # strip .py

            if len(parts) == 1:
                problems = all_problem_names
            else:
                problems = []
                for seg in parts[1].split(","):
                    seg = seg.strip()
                    if not seg:
                        continue
                    if "-" in seg:
                        s, e = seg.split("-", 1)
                        s = s.strip()
                        e = e.strip()
                        if s.isdigit() and e.isdigit():
                            for idx in range(int(s), int(e) + 1):
                                if idx < 1 or idx > len(all_problem_names):
                                    raise ValueError(
                                        f"TBG problem index {idx} out of range (1..{len(all_problem_names)})"
                                    )
                                problems.append(all_problem_names[idx - 1])
                            continue
                    if seg.endswith(".py"):
                        seg = seg[:-3]
                    problems.append(seg)
            tasks.extend({'level': level, 'problem_id': str(p)} for p in problems)
        else:
            level = str(level)
            if len(parts) == 1:
                problem_ids = DATASET_RANGES[test_source][level]
            else:
                problem_ids = []
                for seg in parts[1].split(","):
                    if "-" in seg:
                        s, e = seg.split("-")
                        problem_ids.extend(range(int(s), int(e) + 1))
                    else:
                        problem_ids.append(int(seg))
            tasks.extend({'level': level, 'problem_id': str(p)} for p in problem_ids)
    
    return tasks


def load_config_from_yaml(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Load configuration from YAML file and update args.
    Command line arguments take precedence over YAML values.
    
    Args:
        args: Parsed arguments namespace
        parser: Argument parser instance
        
    Returns:
        Updated arguments namespace
    """
    if args.config is None:
        return args
    
    # Load YAML config file
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Track which arguments were explicitly provided via command line
    # Check sys.argv to see what was provided
    provided_args = set()
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith('--'):
            # Convert --arg-name to arg_name
            arg_name = arg[2:].replace('-', '_')
            provided_args.add(arg_name)
            # Skip the value if it's not a flag (not a store_true/store_false)
            if i + 1 < len(sys.argv):
                next_arg = sys.argv[i + 1]
                # Check if next arg is not another option
                if not next_arg.startswith('-'):
                    # Check if this action takes a value
                    for action in parser._actions:
                        if action.dest == arg_name:
                            # If it's not a store_true/store_false, it takes a value
                            if not isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                                i += 1  # Skip the value
                            break
        i += 1
    
    # Update args with YAML values, but skip those provided via command line
    for key, value in config_dict.items():
        if key in provided_args:
            # This was provided via command line, skip it
            continue
        
        if hasattr(args, key):
            # Find the action to understand the type
            arg_action = None
            for action in parser._actions:
                if action.dest == key:
                    arg_action = action
                    break
            
            # Handle different action types
            if isinstance(arg_action, argparse._StoreTrueAction):
                # For store_true, only set if YAML says True
                if value is True:
                    setattr(args, key, True)
            elif isinstance(arg_action, argparse._StoreFalseAction):
                # For store_false, only set if YAML says False
                if value is False:
                    setattr(args, key, False)
            else:
                # For other types, set the value
                setattr(args, key, value)
    
    return args
