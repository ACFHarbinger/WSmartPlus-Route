"""
Iterated Local Search Post-Processor.
"""

import random
from typing import Any, Dict, List, Union

import numpy as np
import torch

from logic.src.interfaces import IPostProcessor

from .registry import PostProcessorRegistry


@PostProcessorRegistry.register("ils")
class IteratedLocalSearchPostProcessor(IPostProcessor):
    """
    Iterated Local Search (ILS) post-processor.
    """

    def __init__(
        self,
        ls_operator: Union[str, Dict[str, float]] = "2opt",
        perturbation_type: Union[str, Dict[str, float]] = "double_bridge",
        n_restarts: int = 5,
        ls_iterations: int = 50,
        perturbation_strength: float = 0.2,
    ):
        """
        Initialize the ILS post-processor.
        """
        self.ls_operator = ls_operator
        self.perturbation_type = perturbation_type
        self.n_restarts = n_restarts
        self.ls_iterations = ls_iterations
        self.perturbation_strength = perturbation_strength

        self.default_op_probs = {
            "2opt": 0.25,
            "swap": 0.15,
            "relocate": 0.15,
            "two_opt_star": 0.2,
            "swap_star": 0.15,
            "3opt": 0.1,
        }

        self.default_perturb_probs = {
            "double_bridge": 0.5,
            "shuffle": 0.3,
            "random_swap": 0.2,
        }

    def process(self, tour: List[int], **kwargs: Any) -> List[int]:
        """
        Apply ILS to refine the tour.
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not isinstance(distance_matrix, torch.Tensor):
            dm_tensor = torch.from_numpy(np.array(distance_matrix)).float().to(device)
        else:
            dm_tensor = distance_matrix.to(device)

        ops = {
            "2opt": vectorized_two_opt,
            "swap": vectorized_swap,
            "relocate": vectorized_relocate,
            "3opt": vectorized_three_opt,
            "two_opt": vectorized_two_opt,
            "two_opt_star": vectorized_two_opt_star,
            "swap_star": vectorized_swap_star,
            "three_opt": vectorized_three_opt,
        }

        op_probs_dict = None
        ops_sorted: List[str] = []
        op_weights: List[float] = []
        ls_func = None

        if isinstance(self.ls_operator, dict):
            op_probs_dict = self.ls_operator
        elif isinstance(self.ls_operator, str) and self.ls_operator == "random":
            op_probs_dict = self.default_op_probs

        if op_probs_dict:
            ops_sorted = sorted(op_probs_dict.keys())
            op_weights = [op_probs_dict[op] for op in ops_sorted]
        else:
            op_name = self.ls_operator if isinstance(self.ls_operator, str) else "2opt"
            ls_func = ops.get(op_name, vectorized_two_opt)

        p_probs_dict = None
        p_modes_sorted: List[str] = []
        p_weights: List[float] = []

        if isinstance(self.perturbation_type, dict):
            p_probs_dict = self.perturbation_type
        elif isinstance(self.perturbation_type, str) and self.perturbation_type == "random":
            p_probs_dict = self.default_perturb_probs

        if p_probs_dict:
            p_modes_sorted = sorted(p_probs_dict.keys())
            p_weights = [p_probs_dict[m] for m in p_modes_sorted]

        def compute_cost(t: torch.Tensor) -> float:
            """Compute tour cost."""
            if t.dim() == 1:
                t = t.unsqueeze(0)
            from_nodes = t[:, :-1]
            to_nodes = t[:, 1:]
            if dm_tensor.dim() == 3:
                return dm_tensor[0, from_nodes[0], to_nodes[0]].sum().item()
            return dm_tensor[from_nodes, to_nodes].sum().item()

        def perturb(t: torch.Tensor, mode: str) -> torch.Tensor:
            """Apply perturbation to escape local optimum."""
            t_list = t.squeeze(0).tolist()
            n = len(t_list)

            if mode == "double_bridge":
                if n < 8:
                    return t
                positions = sorted(random.sample(range(1, n - 1), 3))
                p1, p2, p3 = positions
                new_list = t_list[:p1] + t_list[p2:p3] + t_list[p1:p2] + t_list[p3:]
                return torch.tensor(new_list, device=device).unsqueeze(0)

            elif mode == "shuffle":
                seg_len = max(2, int(n * self.perturbation_strength))
                if n - seg_len - 1 <= 1:
                    return t
                start = random.randint(1, n - seg_len - 1)
                segment = t_list[start : start + seg_len]
                random.shuffle(segment)
                new_list = t_list[:start] + segment + t_list[start + seg_len :]
                return torch.tensor(new_list, device=device).unsqueeze(0)

            else:
                n_swaps = max(1, int(n * self.perturbation_strength))
                new_list = t_list[:]
                for _ in range(n_swaps):
                    if n < 3:
                        break
                    i, k = random.sample(range(1, n - 1), 2)
                    new_list[i], new_list[k] = new_list[k], new_list[i]
                return torch.tensor(new_list, device=device).unsqueeze(0)

        current = torch.tensor(tour, device=device).unsqueeze(0)

        if op_probs_dict:
            initial_op = random.choices(ops_sorted, weights=op_weights)[0]
            current_ls_func = ops.get(initial_op, vectorized_two_opt)
        else:
            current_ls_func = ls_func if ls_func is not None else vectorized_two_opt

        current = current_ls_func(current, dm_tensor, max_iterations=self.ls_iterations)
        best = current.clone()
        best_cost = compute_cost(best)

        for _ in range(self.n_restarts):
            if p_probs_dict:
                p_mode = random.choices(p_modes_sorted, weights=p_weights)[0]
            else:
                p_mode = self.perturbation_type if isinstance(self.perturbation_type, str) else "double_bridge"

            perturbed = perturb(current, p_mode)

            if op_probs_dict:
                iter_op = random.choices(ops_sorted, weights=op_weights)[0]
                iter_ls_func = ops.get(iter_op, vectorized_two_opt)
            else:
                iter_ls_func = ls_func if ls_func is not None else vectorized_two_opt

            refined = iter_ls_func(perturbed, dm_tensor, max_iterations=self.ls_iterations)
            refined_cost = compute_cost(refined)

            if refined_cost < best_cost - 1e-6:
                best = refined.clone()
                best_cost = refined_cost

            current = refined

        return best.squeeze(0).cpu().tolist()
