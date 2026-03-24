"""
LKH (Lin-Kernighan-Helsgaun) Post-Processor.
"""

from typing import Any, List

import numpy as np
import torch

from logic.src.interfaces import IPostProcessor
from logic.src.policies.other.operators.heuristics.lin_kernighan_helsgaun import solve_lkh

from .base import PostProcessorRegistry


@PostProcessorRegistry.register("lkh")
class LinKernighanHelsgaunPostProcessor(IPostProcessor):
    """
    Refines tours using the Lin-Kernighan-Helsgaun heuristic for TSP.

    Handles VRPP subset routes by extracting a dense sub-matrix containing
    only the visited nodes, running LKH on this sub-problem, and mapping
    the results back to the original node IDs.
    """

    def process(self, tour: List[int], **kwargs: Any) -> List[int]:
        """
        Apply LKH refinement to a tour using sub-matrix extraction.

        For VRPP instances where only a subset of nodes are visited, this
        method creates a dense sub-problem by extracting only the relevant
        rows and columns from the full distance matrix. This prevents index
        out-of-bounds errors and eliminates performance bottlenecks from
        hallucinated nodes during tour merging.

        Args:
            tour: The initial tour to refine (list of node IDs from the full problem).
            **kwargs: Must contain 'distance_matrix'. Optionally 'max_iterations' and 'seed'.

        Returns:
            List[int]: The refined tour with original node IDs.
        """
        # 1. Early Exits & Edge Cases
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None:
            return tour

        if isinstance(distance_matrix, torch.Tensor):
            distance_matrix = distance_matrix.cpu().numpy()
        elif not isinstance(distance_matrix, np.ndarray):
            distance_matrix = np.array(distance_matrix)

        if not tour:
            return tour

        # 2. Ensure Closed Tour Format
        if tour[0] != 0:
            tour = [0] + tour
        if tour[-1] != 0:
            tour = tour + [0]

        # Early exit for trivially small routes (3 or fewer unique nodes)
        # These cannot be improved by k-opt moves
        unique_count = len(set(tour))
        if unique_count <= 3:
            return tour

        # 3. Node Mapping (Core Fix for VRPP Subset Routes)
        # Extract unique nodes, ensuring depot (0) is first
        unique_nodes_set = set(tour)
        unique_nodes = [0] + sorted([n for n in unique_nodes_set if n != 0])

        # Create bidirectional mappings between original IDs and dense indices
        node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}

        # 4. Extract the Sub-Matrix
        # Use NumPy advanced indexing to create a dense distance matrix
        # containing only the rows and columns for visited nodes
        sub_matrix = distance_matrix[np.ix_(unique_nodes, unique_nodes)]

        # 5. Translate the Initial Tour to Dense Indices
        # Map each original node ID in the tour to its dense index
        sub_tour = [node_to_idx[node] for node in tour]

        # 6. Execute LKH on the Sub-Problem
        # Default max_k=3 for VRPP subsets to prevent massive slowdowns
        max_iterations = kwargs.get("max_iterations", self.config.get("max_iterations", 1000))
        max_k = kwargs.get("max_k", self.config.get("max_k", 3))
        seed = kwargs.get("seed", self.config.get("seed", 42))

        optimized_sub_tour, _ = solve_lkh(
            sub_matrix,
            initial_tour=sub_tour,
            max_iterations=max_iterations,
            max_k=max_k,
            seed=seed,
        )

        # 7. Reverse Map the Optimized Tour
        # Convert dense indices back to original VRPP node IDs
        optimized_tour = [unique_nodes[idx] for idx in optimized_sub_tour]

        return optimized_tour
