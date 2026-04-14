"""
Branch-and-Bound tree management for VRPP.
"""

import heapq
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from logic.src.policies.other.branching_solvers.common.node import BranchNode

if TYPE_CHECKING:
    from logic.src.policies.other.branching_solvers.master_problem.model import VRPPMasterProblem


class BranchAndBoundTree:
    """
    Manages the search tree and node frontier for Branch-and-Price-and-Cut.

    Implements the global search control logic, maintaining the frontier of
    unexplored B&B nodes and tracking the global upper and lower bounds.
    Supports several traversal strategies (Best-First, DFS).

    Theoretical Context:
    The tree utilizes the 'best-bound' priority for Best-First search to
    minimize the number of nodes explored before proving optimality. This
    is particularly effective for VRPP where the root LP gap is often
    small (~1-3%), allowing for early pruning.
    """

    def __init__(
        self,
        v_model: Optional["VRPPMasterProblem"] = None,
        params: Optional[Any] = None,
        max_nodes: int = 1000,
        strategy: str = "edge",
        search_strategy: str = "best-first",
    ) -> None:
        """
        Initialise the tree.

        Args:
            v_model: Master problem model.
            params: B&B parameters.
            max_nodes: Maximum number of nodes to explore.
            strategy: Branching strategy.
            search_strategy: Node selection strategy ('best-first', 'dfs').
        """
        self.root = BranchNode()
        self.search_strategy = search_strategy
        self.max_nodes = max_nodes
        self.strategy = strategy
        self.open_nodes: List[BranchNode] = []
        self._counter = 0  # Tie-breaker for heapq

        self.global_ub = float("inf")
        self.global_lb = -float("inf")
        self.best_solution: Optional[Dict[int, float]] = None
        self.best_routes: Optional[List] = None

        # Add root to frontier
        self.push_node(self.root)

    def push_node(self, node: BranchNode) -> None:
        """Add a node to the search frontier."""
        if self.search_strategy == "best-first":
            # Priority queue (negate bound because heapq is a min-heap)
            priority = -(node.lp_bound if node.lp_bound is not None else 1e9)
            heapq.heappush(self.open_nodes, (priority, self._counter, node))
            self._counter += 1
        else:
            # Simple list for DFS (LIFO)
            self.open_nodes.append(node)

    def pop_node(self) -> Optional[BranchNode]:
        """Select and return the next node to explore."""
        if not self.open_nodes:
            return None

        if self.search_strategy == "best-first":
            _, _, node = heapq.heappop(self.open_nodes)
            return node
        else:
            return self.open_nodes.pop()

    def update_global_lb(self, value: float, routes: List, solution: Dict[int, float]) -> bool:
        """
        Update global lower bound (best integer solution found).

        Returns:
            True if a new best solution was found.
        """
        if value > self.global_lb + 1e-6:
            self.global_lb = value
            self.best_routes = routes
            self.best_solution = solution
            logging.info(f"New global LB: {self.global_lb:.4f}")
            return True
        return False

    def prune_nodes(self, incumbent_lb: float) -> int:
        """
        Remove open nodes whose LP bound is worse than the current incumbent.

        Returns:
            Number of nodes pruned.
        """
        initial_count = len(self.open_nodes)
        if self.search_strategy == "best-first":
            # Filter heap and rebuild
            self.open_nodes = [item for item in self.open_nodes if -item[0] > incumbent_lb + 1e-6]
            heapq.heapify(self.open_nodes)
        else:
            self.open_nodes = [n for n in self.open_nodes if (n.lp_bound or 1e9) > incumbent_lb + 1e-6]

        pruned = initial_count - len(self.open_nodes)
        if pruned > 0:
            logging.info(f"Pruned {pruned} nodes from frontier.")
        return pruned

    def get_statistics(self) -> Dict[str, Any]:
        """Return diagnostic statistics about the search progress."""
        return {
            "open_nodes": len(self.open_nodes),
            "global_ub": self.global_ub,
            "global_lb": self.global_lb,
            "gap": (self.global_ub - self.global_lb) / (abs(self.global_lb) + 1e-10),
        }
