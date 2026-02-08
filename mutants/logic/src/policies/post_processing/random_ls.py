"""
Random Local Search Post-Processor.
"""

from typing import Any, List

import numpy as np
import torch
from logic.src.interfaces import IPostProcessor

from .registry import PostProcessorRegistry


@PostProcessorRegistry.register("random")
class RandomLocalSearchPostProcessor(IPostProcessor):
    """
    Performs stochastic local search refinement by applying random operators.
    Mirrors the logic of RandomLocalSearchPolicy but as a post-processor.
    """

    def process(self, tour: List[int], **kwargs: Any) -> List[int]:
        """
        Apply random local search operators stochastically.

        Args:
            tour: The initial tour to refine.
            **kwargs: Context containing 'distance_matrix', 'n_iterations', and optionally 'op_probs'.

        Returns:
            List[int]: The refined tour.
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
        if distance_matrix is None or len(tour) < 4:
            return tour

        n_iterations = kwargs.get("n_iterations", kwargs.get("post_process_iterations", 50))
        op_probs = kwargs.get("op_probs") or {
            "two_opt": 0.25,
            "swap": 0.15,
            "relocate": 0.15,
            "two_opt_star": 0.2,
            "swap_star": 0.15,
            "three_opt": 0.1,
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not isinstance(distance_matrix, torch.Tensor):
            dm_tensor = torch.from_numpy(np.array(distance_matrix)).float().to(device)
        else:
            dm_tensor = distance_matrix.to(device)

        current_routes = torch.tensor(tour, device=device).unsqueeze(0)  # (1, N)

        op_map = {
            "two_opt": vectorized_two_opt,
            "swap": vectorized_swap,
            "relocate": vectorized_relocate,
            "two_opt_star": vectorized_two_opt_star,
            "swap_star": vectorized_swap_star,
            "three_opt": vectorized_three_opt,
        }

        ops_sorted = sorted(op_map.keys())
        probs = torch.tensor([op_probs.get(op, 0.0) for op in ops_sorted], dtype=torch.float32)
        probs = probs / (probs.sum() + 1e-10)

        iter_count = int(n_iterations) if n_iterations is not None else 50
        try:
            op_indices = torch.multinomial(probs, iter_count, replacement=True).tolist()
            for op_idx in op_indices:
                op_name = ops_sorted[op_idx]
                op_func = op_map[op_name]
                current_routes = op_func(current_routes, dm_tensor, max_iterations=1)

            return current_routes.squeeze(0).cpu().tolist()
        except Exception:
            return tour
