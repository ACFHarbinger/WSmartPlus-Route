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
from typing import List, Optional

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

    This is the standard choice for VRP problems as it:
    - Minimizes the number of nodes explored before proving optimality
    - Provides tight bounds early in the search
    - Works well when the LP relaxation is strong

    Characteristics:
    - Time complexity: O(n log n) per selection (due to sorting)
    - Space complexity: O(n) where n is number of open nodes
    - Optimality guarantee: Yes (like all complete B&B strategies)
    - Basis reuse: Poor (jumps across different tree branches)

    Best used for:
    - VRP and VRPP problems with tight LP relaxations
    - Problems where proving optimality quickly is more important than
      finding good feasible solutions early
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

        # Sort by LP bound descending (highest bound first)
        # Nodes without bounds are deprioritized
        open_nodes.sort(key=lambda n: n.lp_bound if n.lp_bound is not None else float("-inf"), reverse=True)

        return open_nodes.pop(0)

    def get_name(self) -> str:
        return "best_first"


class DepthFirstSearch(NodeSelectionStrategy):
    """
    Depth-First Search strategy: selects deepest node in the tree.

    This strategy is recommended for multicommodity flow problems where:
    - LP solves dominate the computational time
    - Reusing the parent's LP basis significantly speeds up child LP solves
    - The branching tree is relatively balanced

    Characteristics:
    - Time complexity: O(n) per selection (simple depth comparison)
    - Space complexity: O(n) where n is number of open nodes
    - Optimality guarantee: Yes (explores entire tree if needed)
    - Basis reuse: Excellent (stays within same branch until fathomed)

    Best used for:
    - ODIMCF (Origin-Destination Integer Multicommodity Flow)
    - Problems where LP solve time dominates pricing time
    - Problems with weak LP relaxations where many nodes must be explored

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

        # Sort by depth descending (deepest first)
        # Break ties by LP bound (highest bound first)
        open_nodes.sort(key=lambda n: (n.depth, n.lp_bound if n.lp_bound is not None else float("-inf")), reverse=True)

        return open_nodes.pop(0)

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
