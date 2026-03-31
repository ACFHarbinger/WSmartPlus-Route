"""
Search strategies for Branch-and-Bound tree node selection in BPC algorithms.

Provides abstraction for different tree exploration strategies:
- Best-First Search (BFS): Explores nodes with best LP bounds first
- Depth-First Search (DFS): Explores deepest nodes first

References:
    - Barnhart, C., Hane, C. A., & Vance, P. H. (1998).
      "Using Branch-and-Price-and-Cut to Solve Origin-Destination Integer
      Multicommodity Flow Problems." Operations Research, 48(2), 318-326.
      (Uses DFS for ODIMCF to leverage LP basis warmstarts)

    - Achterberg, T., Koch, T., & Martin, A. (2005).
      "Branching rules revisited." Operations Research Letters, 33(1), 42-54.
      (Discusses various node selection strategies and their impact)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from ..branch_and_price.branching import BranchNode


class NodeSelectionStrategy(ABC):
    """
    Abstract base class for B&B tree node selection strategies.

    Different strategies prioritize nodes differently, affecting both the
    search performance and the ability to leverage LP basis information.
    """

    @abstractmethod
    def select_node(self, open_nodes: List[BranchNode]) -> Optional[BranchNode]:
        """
        Select and remove the next node to explore from the open node list.

        Args:
            open_nodes: List of unexplored branch nodes

        Returns:
            Selected node, or None if list is empty
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the strategy name for logging and debugging."""
        pass


class BestFirstSearch(NodeSelectionStrategy):
    """
    Best-First Search strategy: selects node with highest LP objective bound.

    While standard for various optimization problems, in the context of BPC
    for VRPP, it can be less efficient than Depth-First Search because it
    frequently jumps across different tree branches, preventing effective
    LP basis reuse.

    Characteristics:
    - Time complexity: O(n log n) per selection (due to sorting)
    - Space complexity: O(n) where n is number of open nodes
    - Optimality guarantee: Yes (like all complete B&B strategies)
    - Basis reuse: Poor (jumps across different tree branches)

    Best used for:
    - Small VRPP instances where tree depth is limited.
    - Situations where proving optimality with the fewest nodes is
      prioritized over computational speed per node.
    """

    def select_node(self, open_nodes: List[BranchNode]) -> Optional[BranchNode]:
        """
        Select node with highest LP bound (maximization problem).

        For maximization problems (VRPP profit maximization), we want
        to explore nodes with the highest upper bound first.

        Args:
            open_nodes: List of unexplored branch nodes

        Returns:
            Node with highest LP bound, or None if list is empty
        """
        if not open_nodes:
            return None

        # Fix 8: Use O(n) scan instead of O(n log n) sort
        def get_best_first_key(i: int) -> float:
            bound = open_nodes[i].lp_bound
            return bound if bound is not None else float("-inf")

        best_idx = max(range(len(open_nodes)), key=get_best_first_key)

        return open_nodes.pop(best_idx)

    def get_name(self) -> str:
        return "best_first"


class DepthFirstSearch(NodeSelectionStrategy):
    """
    Depth-First Search strategy: selects deepest node in the tree.

    This is the preferred default strategy for the BPC solver as it:
    - Maximizes LP basis reuse by staying within the same branch.
    - Achieves significant speedups (often 3-5x) by resolving child LPs
      from the parent's basis using dual simplex.
    - Effectively leverages LP warm-starts as advocated by Barnhart et al. (1998).

    Characteristics:
    - Time complexity: O(n) per selection (linear scan)
    - Space complexity: O(n) where n is number of open nodes
    - Optimality guarantee: Yes (explores entire tree if needed)
    - Basis reuse: Excellent (stays within same branch until fathomed)

    Best used for:
    - High-performance BPC on VRPP where LP solve time dominates.
    - Large instances where memory management of Basis information
      for Best-First search would be prohibitive.

    Implementation Note:
    ----------------------
    When combined with LP warmstarts, DFS can achieve 3-5x speedup on
    large multicommodity flow instances by reusing the basis from the
    parent node (Barnhart et al., 1998, Section 5).

    The solver should leverage Gurobi's or CPLEX's basis warmstart
    capabilities by:
    1. Storing the basis from the parent node's final LP solve
    2. Loading this basis before solving the child node's LP
    3. Using dual simplex to resolve from the warmstart
    """

    def select_node(self, open_nodes: List[BranchNode]) -> Optional[BranchNode]:
        """
        Select deepest node in the tree (highest depth value).

        When multiple nodes have the same depth (siblings), we break ties
        by selecting the node with the highest LP bound to maintain
        some optimality-driven exploration.

        Args:
            open_nodes: List of unexplored branch nodes

        Returns:
            Deepest node, or None if list is empty
        """
        if not open_nodes:
            return None

        # Fix 8: Use O(n) scan instead of O(n log n) sort
        def get_dfs_key(i: int) -> Tuple[int, float]:
            node = open_nodes[i]
            bound = node.lp_bound
            return (node.depth, bound if bound is not None else float("-inf"))

        best_idx = max(range(len(open_nodes)), key=get_dfs_key)

        return open_nodes.pop(best_idx)

    def get_name(self) -> str:
        return "depth_first"


def create_search_strategy(strategy_name: str) -> NodeSelectionStrategy:
    """
    Factory function to create node selection strategies.

    Args:
        strategy_name: Name of the strategy ("best_first" or "depth_first")

    Returns:
        Instance of the requested strategy

    Raises:
        ValueError: If strategy_name is not recognized

    Example:
        >>> strategy = create_search_strategy("best_first")
        >>> next_node = strategy.select_node(open_nodes)
    """
    strategies = {
        "best_first": BestFirstSearch,
        "depth_first": DepthFirstSearch,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown search strategy '{strategy_name}'. Valid options are: {list(strategies.keys())}")

    return strategies[strategy_name]()
