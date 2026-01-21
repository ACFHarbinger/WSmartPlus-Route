"""
Post-processing utilities.
"""
from typing import List

import torch


class EfficiencyOptimizer:
    """Optimizes routes post-generation."""

    def __init__(self, problem, **kwargs):
        self.problem = problem

    def optimize(self, routes: List[torch.Tensor], **kwargs):
        """Refine routes using local search or heuristics."""
        return routes


def decode_routes(actions: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """Convert action sequences to clean route lists."""
    routes = []
    for action in actions:
        route = action[action != 0].tolist()
        routes.append(route)
    return routes


def calculate_efficiency(routes, dist_matrix, demand, capacity):
    """Compute kg/km efficiency metric."""
    return 0.0
