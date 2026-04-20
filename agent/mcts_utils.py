"""
Utility functions for MCTS kernel optimization, including resume functionality.
"""

import os
import json
import logging
from typing import Dict, TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from agent.mcts import MCTSKernelOptimizer, MCTSNode

from src.eval import KernelExecResult

logger = logging.getLogger(__name__)


def load_from_logs(optimizer: 'MCTSKernelOptimizer', log_path: str) -> int:
    """
    Load tree state from existing log files into the optimizer.
    
    Args:
        optimizer: The MCTSKernelOptimizer instance to load into
        log_path: Path to the log directory containing step files
    
    Returns:
        The number of steps loaded (to continue from)
    """
    from agent.mcts import MCTSNode
    
    # Find all step log files
    step_files = []
    for f in os.listdir(log_path):
        if f.startswith("step_") and f.endswith("_log.json"):
            try:
                step_idx = int(f.split("_")[1])
                step_files.append((step_idx, f))
            except ValueError:
                continue
    
    if not step_files:
        logger.warning(f"No step log files found in {log_path}")
        return 0
    
    # Sort by step index
    step_files.sort(key=lambda x: x[0])
    
    # Build node_id -> node mapping
    node_map: Dict[int, 'MCTSNode'] = {}
    step_sequence: List[Tuple[int, int]] = []
    max_step = 0
    loaded_steps = 0
    last_log_data = None
    root_state_from_step0 = None
    
    for step_idx, step_file in step_files:
        # Load log json
        log_file_path = os.path.join(log_path, step_file)
        with open(log_file_path) as f:
            log_data = json.load(f)
        last_log_data = log_data

        node_id = log_data.get("node_id")
        if node_id is None:
            logger.warning(f"Missing node_id in {log_file_path}, skipping step {step_idx}")
            continue

        # Some steps may log an already-existing node (e.g. expansion fallback path).
        # Avoid creating duplicate nodes when node_id already exists.
        if node_id in node_map:
            existing = node_map[node_id]
            existing.visits = log_data.get("visits", existing.visits)
            existing.total_reward = log_data.get("total_reward", existing.total_reward)
            existing.max_reward = log_data.get("max_reward", existing.max_reward)
            step_sequence.append((step_idx, node_id))
            loaded_steps += 1
            max_step = max(max_step, step_idx)
            logger.debug(f"Loaded existing node from step {step_idx}: node_id={node_id}")
            continue
        
        # Load kernel
        kernel_file = os.path.join(log_path, f"step_{step_idx}.py")
        if not os.path.exists(kernel_file):
            logger.warning(f"Kernel file {kernel_file} not found, skipping step {step_idx}")
            continue
        with open(kernel_file) as f:
            kernel = f.read()
        
        # Load metrics
        metrics = _load_metrics(log_path, step_idx)
        if metrics is None:
            logger.warning(f"Metrics file for step {step_idx} not found, skipping")
            continue
        
        # Get parent node
        parent_node_id = log_data.get("parent_node_id")
        parent = node_map.get(parent_node_id) if parent_node_id is not None else None
        if parent_node_id is not None and parent is None:
            logger.warning(
                f"Parent node {parent_node_id} not found when loading step {step_idx}. "
                f"Node {node_id} will be attached without a parent."
            )
        
        # Create node (without using _create_node to avoid side effects)
        node = MCTSNode(
            kernel=kernel,
            metrics=metrics,
            parent=parent,
            node_id=node_id,
            depth=log_data["depth"],
            created_by=log_data["created_by"],
            context_node_ids=log_data.get("context_node_ids", []),
            visits=log_data.get("visits", 1),
            total_reward=log_data.get("total_reward", 0.0),
            max_reward=log_data.get("max_reward", 0.0),
        )
        
        # Add to parent's children
        if parent is not None:
            parent.children.append(node)
        
        # Register node
        node_map[node.node_id] = node
        optimizer.all_nodes.append(node)
        step_sequence.append((step_idx, node.node_id))
        
        # Set root if this is step 0 (dummy root)
        if step_idx == 0:
            optimizer.root = node
            root_state_from_step0 = {
                "visits": log_data.get("visits", 0),
                "total_reward": log_data.get("total_reward", 0.0),
                "max_reward": log_data.get("max_reward", 0.0),
            }
        
        loaded_steps += 1
        max_step = max(max_step, step_idx)
        logger.debug(f"Loaded step {step_idx}: node_id={node.node_id}, created_by={node.created_by}")

    # Fallback root recovery if step_0 is missing but we still loaded nodes.
    if optimizer.root is None and optimizer.all_nodes:
        roots = [n for n in optimizer.all_nodes if n.parent is None]
        if roots:
            optimizer.root = min(roots, key=lambda n: n.depth)
            logger.warning(f"step_0 missing; using node {optimizer.root.node_id} as root")

    # Reconstruct dynamic MCTS statistics by replaying backprop for created nodes.
    # The per-step log stores one node snapshot per step, not a full-tree snapshot.
    if optimizer.root is not None and optimizer.all_nodes:
        for node in optimizer.all_nodes:
            node.visits = 0
            node.total_reward = 0.0
            node.max_reward = 0.0

        if root_state_from_step0 is not None:
            optimizer.root.visits = root_state_from_step0["visits"]
            optimizer.root.total_reward = root_state_from_step0["total_reward"]
            optimizer.root.max_reward = root_state_from_step0["max_reward"]

        replayed_node_ids = set()
        for step_idx, node_id in step_sequence:
            if node_id in replayed_node_ids:
                continue
            replayed_node_ids.add(node_id)
            # Step 0 only records root initialization; do not replay it.
            if step_idx == 0:
                continue
            node = node_map.get(node_id)
            if node is None:
                continue
            optimizer.backpropagate(node, node.reward)

    # Restore global best from the latest step log first (most faithful to pre-resume state).
    global_best_node_id = last_log_data.get("global_best_node_id") if last_log_data else None
    if global_best_node_id in node_map:
        optimizer.global_best_node = node_map[global_best_node_id]
    elif optimizer.all_nodes:
        optimizer.global_best_node = max(optimizer.all_nodes, key=lambda n: n.score)
    else:
        optimizer.global_best_node = None
    
    # Update node counter to avoid ID collision
    if optimizer.all_nodes:
        optimizer.node_counter = max(n.node_id for n in optimizer.all_nodes) + 1
    
    logger.info(
        f"Loaded {loaded_steps} valid steps from {log_path}, max_step={max_step}, "
        f"total_nodes={len(optimizer.all_nodes)}, node_counter={optimizer.node_counter}, "
        f"global_best_node_id={optimizer.global_best_node.node_id if optimizer.global_best_node else None}"
    )
    
    return max_step


def _load_metrics(log_path: str, step_idx: int) -> KernelExecResult:
    """
    Load metrics from file. Supports both JSON and TXT formats.
    
    Args:
        log_path: Path to the log directory
        step_idx: Step index
    
    Returns:
        KernelExecResult or None if not found
    """
    # Try JSON format first (new format)
    json_file = os.path.join(log_path, f"step_{step_idx}_metrics.json")
    if os.path.exists(json_file):
        with open(json_file) as f:
            data = json.load(f)
        return KernelExecResult(
            compiled=data.get("compiled", False),
            correctness=data.get("correctness", False),
            metadata=data.get("metadata", {}),
            runtime=data.get("runtime", -1.0),
            runtime_stats=data.get("runtime_stats", {})
        )
    
    # Try TXT format (old format)
    txt_file = os.path.join(log_path, f"step_{step_idx}_metrics.txt")
    if os.path.exists(txt_file):
        with open(txt_file) as f:
            metrics_str = f.read()
        return _parse_metrics_txt(metrics_str)
    
    return None


def _parse_metrics_txt(metrics_str: str) -> KernelExecResult:
    """
    Parse metrics from old TXT string representation.
    
    Args:
        metrics_str: String representation of metrics
    
    Returns:
        KernelExecResult
    """
    import ast
    
    compiled = "compiled=True" in metrics_str
    correctness = "correctness=True" in metrics_str
    
    # Extract runtime
    runtime = -1.0
    try:
        if "runtime=" in metrics_str:
            runtime_part = metrics_str.split("runtime=")[1]
            runtime_str = runtime_part.split(" ")[0].split("\n")[0]
            runtime = float(runtime_str)
    except (ValueError, IndexError):
        pass
    
    # Extract runtime_stats
    runtime_stats = {}
    try:
        if "runtime_stats=" in metrics_str:
            stats_part = metrics_str.split("runtime_stats=")[1].strip()
            # Find the matching closing brace
            brace_count = 0
            end_idx = 0
            for i, c in enumerate(stats_part):
                if c == '{':
                    brace_count += 1
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            stats_str = stats_part[:end_idx]
            runtime_stats = ast.literal_eval(stats_str)
    except (ValueError, SyntaxError):
        pass
    
    # Extract metadata
    metadata = {}
    try:
        if "metadata=" in metrics_str:
            meta_part = metrics_str.split("metadata=")[1]
            brace_count = 0
            end_idx = 0
            for i, c in enumerate(meta_part):
                if c == '{':
                    brace_count += 1
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            meta_str = meta_part[:end_idx]
            metadata = ast.literal_eval(meta_str)
    except (ValueError, SyntaxError):
        pass
    
    return KernelExecResult(
        compiled=compiled,
        correctness=correctness,
        metadata=metadata,
        runtime=runtime,
        runtime_stats=runtime_stats
    )

