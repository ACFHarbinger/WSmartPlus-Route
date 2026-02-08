"""
Classical Local Search Post-Processor.
"""

from typing import Any, List

import numpy as np
import torch
from logic.src.interfaces import IPostProcessor

from .registry import PostProcessorRegistry


@PostProcessorRegistry.register("classical")
class ClassicalLocalSearchPostProcessor(IPostProcessor):
    """
    Wrapper for vectorized local search operators from
    logic/src/models/policies/classical/local_search.py
    """

    def __init__(self, operator_name: str = "2opt"):
        """
        Initialize the classical local search processor.

        Args:
            operator_name: The name of the local search operator to use
                (e.g., '2opt', 'swap', 'relocate'). Defaults to '2opt'.
        """
        self.operator_name = operator_name

    def process(self, tour: List[int], **kwargs: Any) -> List[int]:
        """
        Apply vectorized local search to the tour.

        Args:
            tour: The initial tour to refine.
            **kwargs: Context containing 'distance_matrix' and optionally 'n_iterations'.

        Returns:
            List[int]: The refined tour after applying the local search operator.
        """
        from logic.src.models.policies.local_search import (
            vectorized_relocate,
            vectorized_swap,
            vectorized_swap_star,
            vectorized_three_opt,
            vectorized_two_opt,
            vectorized_two_opt_star,
        )

        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None:
            return tour

        max_iter = kwargs.get("n_iterations", kwargs.get("post_process_iterations", 50))

        # Ensure Tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not isinstance(distance_matrix, torch.Tensor):
            dm_tensor = torch.from_numpy(np.array(distance_matrix)).float().to(device)
        else:
            dm_tensor = distance_matrix.to(device)

        if len(tour) < 4:
            return tour

        tour_tensor = torch.tensor(tour, device=device).unsqueeze(0)  # (1, N)

        ops = {
            "2opt": vectorized_two_opt,
            "swap": vectorized_swap,
            "relocate": vectorized_relocate,
            "2opt*": vectorized_two_opt_star,
            "swap_star": vectorized_swap_star,
            "3opt": vectorized_three_opt,
            "two_opt": vectorized_two_opt,
            "two_opt_star": vectorized_two_opt_star,
            "three_opt": vectorized_three_opt,
        }

        op_fn = ops.get(self.operator_name, vectorized_two_opt)

        try:
            refined_tensor = op_fn(tour_tensor, dm_tensor, max_iterations=max_iter)
            return refined_tensor.squeeze(0).cpu().tolist()
        except Exception:
            return tour
