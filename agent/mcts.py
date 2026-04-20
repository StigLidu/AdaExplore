# Monte Carlo Tree Search for Kernel Optimization
# Sibling nodes → large steps (propose new kernels)
# Ancestral paths → small refinements (refine existing kernels)

import argparse
import random
import math
import logging
import os
import time
import json
import yaml
import shutil
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm

from agent.actions import single_large_step, single_small_step, dummy_small_step, dummy_large_step
from agent.utils import load_test_source, REPO_TOP_PATH, calculate_score, copy_step_files, read_metrics, dummy_metrics
from agent.inference_server import create_inference_server

logger = logging.getLogger(__name__)

# MCTS Reward Constants
# REWARD_NOT_COMPILED = 0.0  # Reward for kernels that don't compile
# REWARD_COMPILED_BUT_INCORRECT = 0.1  # Reward for kernels that compile but are incorrect
# REWARD_MIN = 0.2  # Minimum reward for correct kernels
# REWARD_MAX = 1.0  # Maximum reward for correct kernels

REWARD_NOT_COMPILED = 0.0
REWARD_COMPILED_BUT_INCORRECT = 0.05
REWARD_MIN = 0.4
REWARD_MAX = 1.6

# Speedup normalization constants
SPEEDUP_MAX = 10.0  # Maximum speedup value for clipping
SPEEDUP_MIN = 0.1  # Minimum speedup value for clipping


@dataclass
class MCTSNode:
    """A node in the MCTS tree representing a kernel state."""
    kernel: str
    metrics: Any  # KernelExecResult
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    
    # MCTS statistics
    visits: int = 0
    total_reward: float = 0.0
    max_reward: float = 0.0
    
    # Metadata
    node_id: int = 0
    depth: int = 0
    created_by: str = "root"  # "large_step" or "small_step"
    context_node_ids: List[int] = field(default_factory=list)  # Context nodes used when creating this node
    prompt: str = ""  # Prompt used to generate this node
    
    # Cache for score (to avoid repeated calculation)
    _score_cache: tuple = field(default=None, repr=False)
    
    @property
    def score(self) -> tuple:
        """Get the score tuple (compiled, correct, speedup).
        Uses cache to avoid repeated calculation.
        """
        if self._score_cache is not None:
            return self._score_cache
        self._score_cache = calculate_score(self.metrics)
        return self._score_cache
    
    @property
    def reward(self) -> float:
        """Calculate reward from cached score for MCTS backpropagation.
        Returns a value in [0, 1] range.
        """
        compiled, correct, speedup = self.score
        
        if not compiled:
            return REWARD_NOT_COMPILED
        if not correct:
            return REWARD_COMPILED_BUT_INCORRECT
        
        # Normalize speedup to [REWARD_MIN, REWARD_MAX] range
        speedup_clipped = max(min(speedup, SPEEDUP_MAX), SPEEDUP_MIN)
        reward_range = REWARD_MAX - REWARD_MIN
        speedup_range = SPEEDUP_MAX - SPEEDUP_MIN
        return min(REWARD_MAX, max(REWARD_MIN, REWARD_MIN + reward_range * (speedup_clipped - SPEEDUP_MIN) / speedup_range))
    
    @property
    def avg_reward(self) -> float:
        """Average reward for this node."""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits
    
    def _blended_reward(self, reward_alpha: float) -> float:
        """Blend max and avg reward: alpha * max + (1 - alpha) * avg."""
        return reward_alpha * self.max_reward + (1 - reward_alpha) * self.avg_reward

    def ucb1(self, exploration_weight: float = 1.414, reward_alpha: float = 1.0) -> float:
        """
        Calculate UCB1 score for node selection.
        UCB1 = reward + exploration_weight * sqrt(ln(parent_visits) / visits)
        
        Args:
            exploration_weight: UCB1 exploration constant
            reward_alpha: Blend coefficient, 1.0 = max_reward, 0.0 = avg_reward
        """
        if self.visits == 0:
            return float('inf')
        
        exploitation = self._blended_reward(reward_alpha)
        
        if self.parent is None or self.parent.visits == 0:
            return exploitation
        
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def expand_ucb1(self, exploration_weight: float = 1.414, reward_alpha: float = 1.0) -> float:
        """
        Calculate UCB1 score for the "expand" action (adding a new child).
        
        Q_value: best blended reward among direct children.
        N_visit: number of existing children.
        
        Returns infinity if no children exist (encourage first expansion).
        """
        if len(self.children) == 0:
            return float('inf')
        
        if exploration_weight <= 0:
            return float('-inf')

        q_value = max(c._blended_reward(reward_alpha) for c in self.children)
        n_expand = len(self.children)

        exploitation = q_value
        if self.visits == 0:
            return exploitation
        exploration = exploration_weight * math.sqrt(math.log(self.visits) / (n_expand ** 2))
        return exploitation + exploration
    
    def should_expand(
        self,
        exploration_weight: float = 1.414,
        reward_alpha: float = 1.0,
        expand_exploration_weight: Optional[float] = None
    ) -> bool:
        """
        Check if this node should be expanded by comparing UCB1 of expand action vs best child.
        Returns True if the expand action has higher UCB1 than selecting the best child.
        """
        if len(self.children) == 0:
            return True

        if expand_exploration_weight is None:
            expand_exploration_weight = exploration_weight

        expand_score = self.expand_ucb1(expand_exploration_weight, reward_alpha)
        best_child_score = max(child.ucb1(exploration_weight, reward_alpha) for child in self.children)
        
        return expand_score > best_child_score

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (no children)."""
        return len(self.children) == 0
    
    def get_path_to_root(self) -> List['MCTSNode']:
        """Get the path from this node to the root."""
        path = [self]
        node = self
        while node.parent is not None:
            path.append(node.parent)
            node = node.parent
        return path[::-1]  # Reverse to get root-to-node path
    
    def get_path_to_cut(self) -> List['MCTSNode']:
        """Get the path from this node to the first ancestor node that is a large step node.
        Since all nodes from dummy_root are created via large_step, this will always find a large_step ancestor
        (unless current node is already a large_step).
        """
        path = [self]
        node = self
        
        # Traverse up until we find a large_step node
        # Since dummy_root's children are always large_step, we will always find one
        # If current node is already large_step, the loop won't execute and we return [self]
        while node.parent is not None and node.created_by != "large_step":
            path.append(node.parent)
            node = node.parent
        
        return path[::-1]  # Reverse to get root-to-node path

class MCTSKernelOptimizer:
    """
    Monte Carlo Tree Search for Kernel Optimization.
    
    Tree structure:
    - Root: Initial state (can be empty or a seed kernel)
    - Siblings: Different kernel proposals (large steps)
    - Parent-child: Refinements of a kernel (small steps)
    """
    
    def __init__(
        self,
        ref_arch_src: str,
        inference_server,
        args: argparse.Namespace,
        log_path: str = None,
    ):
        self.ref_arch_src = ref_arch_src
        self.inference_server = inference_server
        self.args = args
        self.log_path = log_path
        
        # Tree state
        self.root: Optional[MCTSNode] = None
        self.all_nodes: List[MCTSNode] = []
        self.node_counter: int = 0
        
        # Global best tracking
        self.global_best_node: Optional[MCTSNode] = None
        
        # MCTS parameters
        self.exploration_weight = getattr(args, 'exploration_weight', 0.2828)
        self.expand_exploration_ratio = getattr(args, 'expand_exploration_ratio', 1.0)
        self.expand_exploration_weight = self.exploration_weight * self.expand_exploration_ratio
        self.reward_alpha = getattr(args, 'reward_alpha', 1.0)  # α*max + (1-α)*avg in UCB1 (1.0=max, 0.0=avg)
        self.small_step_limit = getattr(args, 'small_step_limit', 2)  # Max number of small steps per node
        
        if log_path is not None:
            os.makedirs(log_path, exist_ok=True)
    
    def _create_node(
        self,
        kernel: str,
        metrics: Any,
        parent: Optional[MCTSNode] = None,
        created_by: str = "root",
        context_node_ids: List[int] = None,
        prompt: str = ""
    ) -> MCTSNode:
        """Create a new node and add it to the tree."""
        if context_node_ids is None:
            context_node_ids = []
        
        if getattr(self.args, 'dummy', False):
            metrics = dummy_metrics()
        node = MCTSNode(
            kernel=kernel,
            metrics=metrics,
            parent=parent,
            node_id=self.node_counter,
            depth=parent.depth + 1 if parent else 0,
            created_by=created_by,
            context_node_ids=context_node_ids,
            prompt=prompt,
        )
        self.node_counter += 1
        self.all_nodes.append(node)
        
        if parent is not None:
            parent.children.append(node)
        
        # Update global best using full score tuple ordering:
        # (compiled, correctness, speedup)
        if self.global_best_node is None or node.score > self.global_best_node.score:
            self.global_best_node = node
            logger.debug(f"New global best: node {node.node_id} with score {node.score}")
        
        return node
    
    def select(self) -> MCTSNode:
        """
        Selection phase: Select a node to expand using UCB1.
        Traverse from root to a leaf, always choosing the child with highest UCB1.
        """
        if self.root is None:
            raise ValueError("Tree not initialized. Call initialize() first.")
        
        node = self.root
        alpha = self.reward_alpha
        while not node.is_leaf():
            best_child = max(node.children, key=lambda c: c.ucb1(self.exploration_weight, alpha))
            if node.should_expand(self.exploration_weight, alpha, self.expand_exploration_weight):
                expand_score = node.expand_ucb1(self.expand_exploration_weight, alpha)
                best_child_score = max(c.ucb1(self.exploration_weight, alpha) for c in node.children)
                logger.debug(
                    f"Select: node {node.node_id} expanding: expand_score={expand_score:.4f}, "
                    f"best_child_score={best_child_score:.4f}"
                )
                break
            else:
                node = best_child

        return node
    
    def _get_small_step_component(self, start_node: MCTSNode) -> List[MCTSNode]:
        """
        Get all nodes in the small_step connected component starting from start_node.
        This includes all nodes reachable from start_node via small_step edges (including branches).
        
        Args:
            start_node: The large_step node to start from
        
        Returns:
            List of all nodes in the small_step component (including start_node itself)
        """
        assert start_node.created_by == "large_step", \
            f"start_node must be created by large_step, got {start_node.created_by}"
        component = []
        queue = [start_node]
        visited = set()
        
        while queue:
            current = queue.pop(0)
            if current.node_id in visited:
                continue
            visited.add(current.node_id)
            component.append(current)
            
            # Add all children that are reachable via small_step
            for child in current.children:
                if child.created_by == "small_step" and child.node_id not in visited:
                    queue.append(child)
        
        return component
    
    def _get_diverse_pool_for_large_step(self, node: MCTSNode) -> List[MCTSNode]:
        """
        Get a diverse pool of kernels for large_step expansion by:
        1. Consider path from current node to root
        2. For each small_step connected component (starting from a large_step node, 
           including all branches), only select the best kernel from that component
        3. If more nodes than pool_size, select the best ones; if fewer, don't fill
        
        This ensures diversity by selecting at most one kernel from each small_step branch.
        
        Args:
            node: The current node from which to build the diverse pool
        
        Returns:
            List of diverse nodes for the pool (may be less than pool_size)
        """
        pool_size = getattr(self.args, 'pool_size', 5)
        
        # Step 1: Get path from current node to root
        path_to_root = node.get_path_to_root()
        
        # Step 2: Identify large_step nodes in the path (these are branch points)
        # Note: dummy_root is not a branch point since all its children are large_step
        branch_points = [n for n in path_to_root if n.created_by == "large_step"]
        
        # Step 3: For each branch point, find all nodes in its small_step connected component
        # and select only the best one from each component
        # Note: Each large_step node starts a disjoint component (separated by large_step boundaries),
        selected_nodes = []
        
        for branch_node in branch_points:
            # Get all nodes in the small_step component starting from this branch point
            # This includes the branch_node itself and all nodes reachable via small_step
            component_nodes = self._get_small_step_component(branch_node)
            
            # Filter to only correct kernels
            correct_component_nodes = [n for n in component_nodes if n.score[1]]
            
            # Select only the best kernel from this component
            if correct_component_nodes:
                best_node = max(correct_component_nodes, key=lambda n: n.score)
                selected_nodes.append(best_node)
        
        # Step 4: If we have more nodes than pool_size, randomly select pool_size nodes
        # If we have fewer, don't fill (keep it diverse)
        if len(selected_nodes) > pool_size:
            selected_nodes = random.sample(selected_nodes, pool_size)

        # Step 5: If we have fewer than pool_size nodes,
        # Consider other components beyond the path to root, collect the best nodes from each, and use softmax to get the probability distribution
        # Use a Bernoulli distribution to determine the total number of the extra nodes to select, max = min(pool_size - len(selected_nodes), args.pool_size_extra_max)
        if len(selected_nodes) < pool_size:
            pool_size_extra_max = getattr(self.args, 'pool_size_extra_max', 3)
            max_extra_nodes = min(pool_size - len(selected_nodes), pool_size_extra_max)
            
            # Find all large_step nodes not in the path to root
            path_node_ids = set(n.node_id for n in path_to_root)
            other_large_step_nodes = [
                n for n in self.all_nodes 
                if n.created_by == "large_step" and n.node_id not in path_node_ids
            ]
            
            # For each, get the best node from its small_step component
            # Note: Components are disjoint (separated by large_step boundaries), no overlap possible
            candidate_nodes = []
            for large_node in other_large_step_nodes:
                component_nodes = self._get_small_step_component(large_node)
                correct_nodes = [n for n in component_nodes if n.score[1]]
                if correct_nodes:
                    best_node = max(correct_nodes, key=lambda n: n.score)
                    candidate_nodes.append(best_node)
            
            # If we have candidate nodes, use softmax to sample
            if candidate_nodes and max_extra_nodes > 0:
                # Get cached scores - use speedup as the value for softmax
                scores = [n.score for n in candidate_nodes]
                # Use speedup (index 2) for softmax; 0 if not correct (shouldn't happen as we filtered)
                speedups = [s[2] if s[1] else 0.0 for s in scores]
                
                # Apply softmax with temperature to get probability distribution
                softmax_temperature = self.args.softmax_temperature
                speedups_arr = np.array(speedups, dtype=np.float64)
                # Numerical stability: subtract max before exp
                exp_scores = np.exp((speedups_arr - speedups_arr.max()) / softmax_temperature)
                probs = exp_scores / exp_scores.sum()
                
                # Use truncated geometric distribution to determine how many extra nodes to select
                # P(X=k) ∝ (1-p)^k for k = 0, 1, ..., max_extra_nodes, then normalize
                # Higher p means more likely to select fewer nodes
                geometric_p = self.args.geometric_p
                k_values = np.arange(max_extra_nodes + 1)
                unnorm_probs = (1 - geometric_p) ** k_values
                geom_probs = unnorm_probs / unnorm_probs.sum()
                num_extra = np.random.choice(max_extra_nodes + 1, p=geom_probs)
                
                # Sample nodes according to softmax probabilities
                if num_extra > 0:
                    num_to_select = min(num_extra, len(candidate_nodes))
                    selected_indices = np.random.choice(
                        len(candidate_nodes),
                        size=num_to_select,
                        replace=False,
                        p=probs
                    )
                    for idx in selected_indices:
                        selected_nodes.append(candidate_nodes[idx])
        
        return selected_nodes
    
    def expand_large(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Expansion via large step: Create a new child node.
        This proposes a completely new kernel using a diverse pool of best kernels.
        The pool is selected to ensure diversity by choosing at most one kernel from each small_step branch.
        """
        # Get diverse pool of kernels
        selected_nodes = self._get_diverse_pool_for_large_step(node)
        selected_kernels = [n.kernel for n in selected_nodes]
        selected_metrics = [n.metrics for n in selected_nodes]
        context_node_ids = [n.node_id for n in selected_nodes]

        logger.debug(f"Large step expansion from node {node.node_id}, depth {node.depth}")
        if getattr(self.args, 'dummy', False):
            fn = dummy_large_step
        else:
            fn = single_large_step
        proposal_kernel, proposal_metrics, logs = fn(
            self.ref_arch_src,
            self.inference_server,
            selected_kernels,
            selected_metrics,
            self.args,
        )
        
        # Extract prompt from logs
        prompt = logs.get("prompt", "")
        
        # Add as child node (same as small step)
        new_node = self._create_node(
            kernel=proposal_kernel,
            metrics=proposal_metrics,
            parent=node,
            created_by="large_step",
            context_node_ids=context_node_ids,
            prompt=prompt
        )
        
        return new_node
    
    def expand_small(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Expansion via small step: Create a child node by refining the current kernel.
        This follows the ancestral path for refinement.
        """
        # Get ancestral path for context
        path = node.get_path_to_cut()
        max_memory = getattr(self.args, 'max_memory_round', 5)
        
        # Use recent ancestors as context
        context_nodes = path[-max_memory:] if len(path) > max_memory else path
        context_kernels = [n.kernel for n in context_nodes if n.kernel]
        context_metrics = [n.metrics for n in context_nodes if n.metrics]
        context_node_ids = [n.node_id for n in context_nodes]
        
        logger.debug(f"Small step expansion from node {node.node_id}, depth {node.depth}")
        if getattr(self.args, 'dummy', False):
            fn = dummy_small_step
        else:
            fn = single_small_step
        refined_kernel, refined_metrics, logs = fn(
            self.ref_arch_src,
            self.inference_server,
            context_kernels,
            context_metrics,
            self.args,
        )
        
        # Extract prompt from logs
        prompt = logs.get("prompt", "")
        
        new_node = self._create_node(
            kernel=refined_kernel,
            metrics=refined_metrics,
            parent=node,
            created_by="small_step",
            context_node_ids=context_node_ids,
            prompt=prompt
        )
        
        return new_node
    
    def simulate(self, node: MCTSNode, num_rollouts: int = 1) -> float:
        """
        Simulation phase: Estimate the value of a node.
        For kernel optimization, we directly use the node's reward.
        Optionally, can do multiple small step rollouts.
        """
        total_reward = node.reward
        
        # Optional: Do additional rollouts with small steps
        if num_rollouts > 1:  # Only if current is correct
            raise NotImplementedError("Multiple small step rollouts are not implemented")
        
        return total_reward / num_rollouts
    
    def backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagation phase: Update statistics along the path to root.
        """
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            if reward > current.max_reward:
                current.max_reward = reward
            current = current.parent
    
    def initialize(self, seed_kernel: str = None, seed_metrics: Any = None):
        """
        Initialize the MCTS tree with a dummy root node.
        If seed_kernel is provided, put it directly on the dummy root.
        Otherwise, dummy root has empty kernel and zero metrics.
        No additional large_step nodes are created during initialization.
        """
        from src.eval import KernelExecResult
        
        # Create dummy root node
        if seed_kernel is not None and seed_metrics is not None:
            # Put seed kernel directly on dummy root
            self.root = self._create_node(
                kernel=seed_kernel,
                metrics=seed_metrics,
                parent=None,
                created_by="dummy_root"
            )
            # Backpropagate dummy root
            self.backpropagate(self.root, self.root.reward)
        else:
            # Create dummy root with empty kernel and zero metrics
            dummy_kernel = ""
            dummy_metrics = KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={},
                runtime=-1.0,
                runtime_stats={}
            )
            self.root = self._create_node(
                kernel=dummy_kernel,
                metrics=dummy_metrics,
                parent=None,
                created_by="dummy_root"
            )
        
        # Save step 0 (dummy root)
        if self.log_path is not None:
            self._save_step_log(0, self.root)
        
        logger.debug(f"Tree initialized with dummy root node")
    
    def step(self, step_idx: int) -> MCTSNode:
        """
        Perform one MCTS iteration:
        1. Select a promising node
        2. Expand it (large or small step)
        3. Simulate/evaluate
        4. Backpropagate
        """
        # Selection
        selected_node = self.select()
        logger.debug(f"Step {step_idx}: Selected node {selected_node.node_id} (depth={selected_node.depth}, visits={selected_node.visits})")
        
        # Decide expansion type based on node state
        # If selected node is dummy root, always use large step
        # Otherwise, use large step if already has small_step_limit+ children created by small_step, or randomly
        if selected_node.created_by == "dummy_root":
            use_large_step = True
        else:
            p_large = getattr(self.args, 'p_large', 0.25)
            num_small_step_children = sum(1 for child in selected_node.children if child.created_by == "small_step")
            use_large_step = (
                num_small_step_children >= self.small_step_limit or
                random.random() < p_large
            )
        
        # Expansion
        if use_large_step:
            new_node = self.expand_large(selected_node)
        else:
            new_node = self.expand_small(selected_node)
        
        if new_node is None:
            # Fallback: try the other expansion type
            if use_large_step:
                new_node = self.expand_small(selected_node)
            else:
                new_node = self.expand_large(selected_node)
        
        if new_node is None:
            logger.warning(f"Step {step_idx}: Both expansion types failed")
            return selected_node
        
        # Simulation
        reward = self.simulate(new_node, num_rollouts=1)
        
        # Backpropagation
        self.backpropagate(new_node, reward)
        
        logger.debug(f"Step {step_idx}: Created node {new_node.node_id} via {new_node.created_by}, reward={reward:.4f}")
        
        return new_node
    
    def run(self, num_iterations: int, start_step: int = 0) -> tuple:
        """
        Run MCTS for a specified number of iterations.
        Returns (best_kernel, best_metrics).
        
        Args:
            num_iterations: Total number of iterations to run
            start_step: Step to start from (for resume functionality)
        """
        if self.root is None:
            self.initialize()
        
        # Calculate remaining iterations
        remaining = num_iterations - start_step
        if remaining <= 0:
            logger.info(f"Already completed {start_step} steps, nothing to do (target: {num_iterations})")
            return self.global_best_node.kernel, self.global_best_node.metrics
        
        for i in tqdm(range(remaining), desc=f"MCTS on problem {self.args.level}_{self.args.problem_id}"):
            step_idx = start_step + i + 1
            new_node = self.step(step_idx)
            
            # Logging
            if self.log_path is not None:
                self._save_step_log(step_idx, new_node)
        
        return self.global_best_node.kernel, self.global_best_node.metrics
    
    def _save_step_log(self, step_idx: int, node: MCTSNode):
        """Save step information to log files."""
        with open(os.path.join(self.log_path, f"step_{step_idx}.py"), "w") as f:
            f.write(node.kernel)
        with open(os.path.join(self.log_path, f"step_{step_idx}_metrics.json"), "w") as f:
            json.dump(node.metrics.to_dict(), f, indent=4)
        
        # Save the prompt alongside per-step artifacts for later analysis.
        if node.prompt:
            with open(os.path.join(self.log_path, f"step_{step_idx}_prompt.txt"), "w") as f:
                f.write(node.prompt)
        
        log_dict = {
            "node_id": node.node_id,
            "parent_node_id": node.parent.node_id if node.parent else None,
            "context_node_ids": node.context_node_ids,
            "depth": node.depth,
            "created_by": node.created_by,
            "visits": node.visits,
            "total_reward": node.total_reward,
            "max_reward": node.max_reward,
            "avg_reward": node.avg_reward,
            "score": list(node.score),
            "total_nodes": len(self.all_nodes),
            "global_best_node_id": self.global_best_node.node_id if self.global_best_node else None,
            "global_best_score": list(self.global_best_node.score) if self.global_best_node else None,
        }
        with open(os.path.join(self.log_path, f"step_{step_idx}_log.json"), "w") as f:
            json.dump(log_dict, f, indent=4)
    
    def get_tree_stats(self) -> Dict[str, Any]:
        """Get statistics about the current tree."""
        if not self.all_nodes:
            return {}
        
        depths = [n.depth for n in self.all_nodes]
        visits = [n.visits for n in self.all_nodes]
        rewards = [n.avg_reward for n in self.all_nodes]
        
        return {
            "total_nodes": len(self.all_nodes),
            "max_depth": max(depths),
            "avg_depth": sum(depths) / len(depths),
            "total_visits": sum(visits),
            "avg_visits": sum(visits) / len(visits),
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
            "large_step_nodes": sum(1 for n in self.all_nodes if n.created_by == "large_step"),
            "small_step_nodes": sum(1 for n in self.all_nodes if n.created_by == "small_step"),
        }


def mcts_search(
    ref_arch_src: str,
    inference_server,
    args: argparse.Namespace,
    log_path: str = None
) -> tuple:
    """
    Main entry point for MCTS kernel optimization.
    Returns (best_kernel, best_metrics).
    
    Args:
        ref_arch_src: Reference architecture source code
        inference_server: Inference server for LLM queries
        args: Arguments namespace
        log_path: Path to save logs
    """
    from agent.mcts_utils import load_from_logs
    
    optimizer = MCTSKernelOptimizer(
        ref_arch_src=ref_arch_src,
        inference_server=inference_server,
        args=args,
        log_path=log_path,
    )
    
    total_steps = getattr(args, 'total_steps', 25)
    start_step = 0
    
    # Resume from existing logs if specified
    if args.resume_from is not None:
        start_step = load_from_logs(optimizer, args.resume_from)
        logger.info(f"Resumed from {args.resume_from}, loaded {start_step} steps, will continue from step {start_step + 1}")
        
        # Copy existing files to new log_path if different
        if log_path and log_path != args.resume_from:
            copy_step_files(args.resume_from, log_path)
    
    best_kernel, best_metrics = optimizer.run(total_steps, start_step=start_step)
    
    # Log final statistics
    stats = optimizer.get_tree_stats()
    logger.info(f"MCTS completed. Stats: {stats}")
    
    if log_path is not None:
        with open(os.path.join(log_path, "tree_stats.json"), "w") as f:
            json.dump(stats, f, indent=4)
    
    # Print tree structure and paths in debug mode
    def print_tree(node, indent=0, file=None):
        """
        Recursively print the MCTS tree structure. Each point displays node_id, metric, created_by, max_reward, visits, ucb, context.
        If file is provided, also writes to the file.
        """
        metric_str = (
            f"compiled={getattr(node.metrics, 'compiled', None)}, "
            f"correctness={getattr(node.metrics, 'correctness', None)}, "
            f"runtime={getattr(node.metrics, 'runtime', None)}"
        )
        ucb_value = node.ucb1(optimizer.exploration_weight, optimizer.reward_alpha)
        context_str = f"context=[{', '.join(map(str, node.context_node_ids))}]" if node.context_node_ids else "context=[]"
        line = "    " * indent + f"[{node.node_id}] {node.created_by} - {metric_str} | node_reward={node.reward:.4f}, max_reward={node.max_reward:.4f}, visits={node.visits}, ucb={ucb_value:.4f}, {context_str}"
        logger.debug(line)
        if file is not None:
            file.write(line + "\n")
        for child in node.children:
            print_tree(child, indent + 1, file)

    def print_paths(node, path=None, step_types=None):
        """
        Print the paths from the root to all leaf nodes, each segment注明是large_step还是small_step
        """
        if path is None:
            path = []
        if step_types is None:
            step_types = []
        path.append(node)
        if node.parent is not None:
            step_types.append(node.created_by)
        else:
            step_types.append("root")
        if not node.children:
            # Print the path
            path_info = []
            for idx, n in enumerate(path):
                m = n.metrics
                metric_str = (
                    f"(id={n.node_id}, by={n.created_by}, "
                    f"compiled={getattr(m, 'compiled', None)}, "
                    f"correctness={getattr(m, 'correctness', None)}, "
                    f"runtime={getattr(m, 'runtime', None)})"
                )
                step_type = step_types[idx] if idx < len(step_types) else ""
                if idx > 0:
                    metric_str = step_type + " -> " + metric_str
                path_info.append(metric_str)
            logger.debug("Path: " + " -> ".join(path_info))
        else:
            for child in node.children:
                print_paths(child, path.copy(), step_types.copy())

    if optimizer.root is not None:
        logger.debug("========== MCTS Tree Structure ==========")
        # Save tree structure to file if log_path is provided
        file_handle = None
        if log_path is not None:
            tree_structure_path = os.path.join(log_path, "tree_structure.txt")
            file_handle = open(tree_structure_path, "w")
            file_handle.write("========== MCTS Tree Structure ==========\n")
        
        print_tree(optimizer.root, file=file_handle)
        
        if file_handle is not None:
            file_handle.close()
        
        #logger.debug("========== MCTS All Root-to-Leaf Paths ==========")
        #print_paths(optimizer.root)
    else:
        logger.debug("Cannot output MCTS tree structure: no tree root node found.")
    
    return best_kernel, best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Test Configs
    parser.add_argument("--test_source", type=str, default="KB", choices=["KB", "SYN"])
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--problem_id", type=int, default=1)
    parser.add_argument("--test_list_path", type=str, default=None)
    parser.add_argument("--dtype_str", type=str, default="fp32")
    parser.add_argument("--gpu_name", type=str, default="A6000")
    parser.add_argument("--gpu_architecture", type=str, default="Ampere")
    parser.add_argument("--gpu_id", type=int, default=0)

    # Base Model Configs
    parser.add_argument("--server_type", type=str, default="azure", choices=["azure", "openai", "claude"])
    parser.add_argument("--model_name", type=str, default="gpt-5-mini")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_completion_tokens", type=int, default=16384)
    parser.add_argument("--general_memory_path", type=str, default=None)

    # MCTS Configs
    parser.add_argument("--total_steps", type=int, default=25)
    parser.add_argument("--max_memory_round", type=int, default=5)
    parser.add_argument("--pool_size", type=int, default=5)
    parser.add_argument("--pool_size_extra_max", type=int, default=3)  # Max extra nodes from other components
    parser.add_argument("--softmax_temperature", type=float, default=1.0)  # Temperature for softmax sampling
    parser.add_argument("--geometric_p", type=float, default=0.5)  # Parameter for truncated geometric distribution (higher = fewer extra nodes)
    parser.add_argument("--disable_reviewer", action="store_true", default=False)
    parser.add_argument("--exploration_weight", type=float, default=0.25)  # UCB1 exploration constant
    parser.add_argument("--expand_exploration_ratio", type=float, default=1.0)  # Scale factor for expand-action exploration
    parser.add_argument("--p_large", type=float, default=0.25)  # Probability of large step
    parser.add_argument("--reward_alpha", type=float, default=1.0)  # α*max + (1-α)*avg in UCB1 (1.0=max, 0.0=avg)
    parser.add_argument("--small_step_limit", type=int, default=2)  # Max number of small steps per node
    parser.add_argument("--dummy", action="store_true", default=False)  # Whether to use dummy

    # Logging Configs
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # Configure logging level
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    torch.cuda.set_device(args.gpu_id)
    start_time = time.strftime("%m%d-%H%M%S", time.localtime())
    
    try:
        memory_version = args.general_memory_path.split("_")[2].split(".")[0]
    except:
        memory_version = "none"

    inference_server = create_inference_server(server_type=args.server_type)
    
    if args.save_path is None:
        args.save_path = os.path.join(REPO_TOP_PATH, "outputs", f"MCTS_{memory_version}_{args.test_source}_{args.total_steps}_{start_time}")
    os.makedirs(args.save_path, exist_ok=True)
    
    with open(os.path.join(args.save_path, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f, sort_keys=False)

    if args.test_list_path is None:
        # Test on a single problem
        result_save_path = os.path.join(args.save_path, f"{args.level}_{args.problem_id}")
        os.makedirs(result_save_path, exist_ok=True)
        problem_name, ref_arch_src = load_test_source(args.test_source, args.level, args.problem_id)
        
        global_best_kernel, global_best_metrics = mcts_search(
            ref_arch_src, inference_server, args, log_path=result_save_path
        )
        
        #logger.info(f"Global Best Kernel: {global_best_kernel}")
        logger.info(f"Global Best Metrics: {global_best_metrics}")
        
        with open(os.path.join(result_save_path, "global_best_kernel.py"), "w") as f:
            f.write(global_best_kernel)
        with open(os.path.join(result_save_path, "global_best_metrics.txt"), "w") as f:
            f.write(str(global_best_metrics))
    else:
        # Test on a list of problems
        with open(args.test_list_path, "r") as f:
            test_list = f.readlines()
        test_list = [line.strip() for line in test_list]
        
        correct_count = 0
        sum_speedup = 0
        total_count = 0
        
        for test_problem in tqdm(test_list):
            total_count += 1
            test_level, test_problem_id = (
                int(test_problem.split(" ")[0].strip()),
                int(test_problem.split(" ")[1].strip()),
            )
            args.level = test_level
            args.problem_id = test_problem_id
            result_save_path = os.path.join(args.save_path, f"{test_level}_{test_problem_id}")
            
            if os.path.exists(result_save_path):
                if "global_best_kernel.py" in os.listdir(result_save_path) and "global_best_metrics.txt" in os.listdir(result_save_path):
                    correctness, fast_p = read_metrics(os.path.join(result_save_path, "global_best_metrics.txt"))
                    if correctness:
                        correct_count += 1
                        sum_speedup += fast_p
                    continue
                else:
                    shutil.rmtree(result_save_path)
            
            os.makedirs(result_save_path, exist_ok=True)
            
            logger.debug(f"Testing problem level {test_level} problem id {test_problem_id}")
            problem_name, ref_arch_src = load_test_source(args.test_source, test_level, test_problem_id)
            
            global_best_kernel, global_best_metrics = mcts_search(
                ref_arch_src, inference_server, args, log_path=result_save_path
            )
            
            if global_best_metrics.correctness:
                correct_count += 1
                sum_speedup += global_best_metrics.runtime_stats["fast_p"]
            
            with open(os.path.join(result_save_path, "global_best_kernel.py"), "w") as f:
                f.write(global_best_kernel)
            with open(os.path.join(result_save_path, "global_best_metrics.txt"), "w") as f:
                f.write(str(global_best_metrics))
        
        logger.info(f"Correct count: {correct_count}, Sum speedup: {sum_speedup}, Total count: {total_count}")
        if total_count > 0:
            logger.info(f"Average speedup: {sum_speedup / total_count}")
            logger.info(f"Accuracy: {correct_count / total_count}")
