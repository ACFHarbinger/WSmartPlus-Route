"""
GP Tree definitions for the Genetic Programming Hyper-Heuristic (GPHH) solver.

This module implements **Pure Arithmetic Genetic Programming** where every node
evaluates to a continuous float value, satisfying the Closure Property.

The root node's float output is later mapped to a discrete LLH index in solver.py
using modulo arithmetic.
"""

import random
from typing import Dict, Tuple, Union


class TerminalNode:
    """
    GP terminal node: extracts a continuous feature from the routing context.

    Attributes:
        feature: Name of the feature to extract from the context dictionary.
    """

    def __init__(self, feature: str):
        self.feature = feature

    def evaluate(self, ctx: Dict[str, float]) -> float:
        """
        Evaluate the terminal by extracting its feature value.

        Args:
            ctx: Feature context dictionary.

        Returns:
            Float value of the feature (default 0.0 if not found).
        """
        return ctx.get(self.feature, 0.0)

    def copy(self) -> "TerminalNode":
        """Create a deep copy of this terminal node."""
        return TerminalNode(self.feature)

    def size(self) -> int:
        """
        Return the size of this terminal node.

        Returns:
            Always returns 1 (single node).
        """
        return 1


class FunctionNode:
    """
    GP function node: applies an arithmetic operation to two child subtrees.

    All operations return continuous float values, maintaining the Closure Property.

    Attributes:
        fn: Name of the arithmetic function (ADD, SUB, MUL, DIV, MAX, MIN).
        left: Left child subtree.
        right: Right child subtree.
    """

    def __init__(self, fn: str, left: "GPNode", right: "GPNode"):
        self.fn = fn
        self.left = left
        self.right = right

    def evaluate(self, ctx: Dict[str, float]) -> float:
        """
        Recursively evaluate the arithmetic function.

        Args:
            ctx: Feature context dictionary passed to children.

        Returns:
            Float result of the arithmetic operation.
        """
        left_val = self.left.evaluate(ctx)
        right_val = self.right.evaluate(ctx)

        if self.fn == "ADD":
            return left_val + right_val
        elif self.fn == "SUB":
            return left_val - right_val
        elif self.fn == "MUL":
            return left_val * right_val
        elif self.fn == "DIV":
            # Protected division: return 1.0 if denominator is near-zero
            return left_val / right_val if abs(right_val) > 1e-9 else 1.0
        elif self.fn == "MAX":
            return max(left_val, right_val)
        elif self.fn == "MIN":
            return min(left_val, right_val)
        else:
            return 0.0  # Fallback for unknown functions

    def copy(self) -> "FunctionNode":
        """Create a deep copy of this function node and its subtrees."""
        return FunctionNode(
            self.fn,
            self.left.copy(),
            self.right.copy(),
        )

    def size(self) -> int:
        """
        Return the size of this function node and its subtrees.

        Recursively computes the total number of nodes in the tree rooted at
        this function node by summing 1 (for this node) plus the sizes of the
        left and right subtrees.

        Returns:
            Total number of nodes in the subtree (1 + left.size() + right.size()).
        """
        return 1 + self.left.size() + self.right.size()


GPNode = Union[TerminalNode, FunctionNode]

# Available terminal features extracted from routing state
_TERMINALS = ["avg_node_profit", "load_factor", "route_count", "iter_progress"]

# Available arithmetic functions (all return float)
_FUNCTIONS = ["ADD", "SUB", "MUL", "DIV", "MAX", "MIN"]


def _random_tree(depth: int, n_llh: int, rng: random.Random) -> GPNode:
    """
    Generate a random GP tree with pure arithmetic structure.

    Args:
        depth: Maximum depth of the tree (0 = terminal only).
        n_llh: Number of low-level heuristics (not used in pure arithmetic, kept for API compatibility).
        rng: Random number generator.

    Returns:
        Root node of the generated tree.
    """
    if depth == 0 or rng.random() < 0.4:
        return TerminalNode(rng.choice(_TERMINALS))

    fn = rng.choice(_FUNCTIONS)
    return FunctionNode(
        fn,
        _random_tree(depth - 1, n_llh, rng),
        _random_tree(depth - 1, n_llh, rng),
    )


def _subtree_crossover(t1: GPNode, t2: GPNode, rng: random.Random) -> Tuple[GPNode, GPNode]:
    """
    Perform single-point subtree swap crossover between two GP trees.

    Args:
        t1: First parent tree.
        t2: Second parent tree.
        rng: Random number generator.

    Returns:
        Tuple of two offspring trees.
    """
    c1 = t1.copy()
    c2 = t2.copy()

    # If both are function nodes, swap left or right subtrees
    if isinstance(c1, FunctionNode) and isinstance(c2, FunctionNode):
        if rng.random() < 0.5:
            c1.left, c2.left = c2.left, c1.left
        else:
            c1.right, c2.right = c2.right, c1.right
    return c1, c2


def _mutate(tree: GPNode, depth: int, n_llh: int, rng: random.Random) -> GPNode:
    """
    Mutate a GP tree by replacing a random subtree with a new random tree.

    Args:
        tree: Tree to mutate.
        depth: Maximum depth for new random subtrees.
        n_llh: Number of low-level heuristics (kept for API compatibility).
        rng: Random number generator.

    Returns:
        Mutated tree.
    """
    if isinstance(tree, FunctionNode) and rng.random() < 0.5:
        # Mutate either left or right subtree
        if rng.random() < 0.5:
            tree.left = _random_tree(depth - 1, n_llh, rng)
        else:
            tree.right = _random_tree(depth - 1, n_llh, rng)
        return tree
    # Replace entire tree with new random tree
    return _random_tree(depth, n_llh, rng)
