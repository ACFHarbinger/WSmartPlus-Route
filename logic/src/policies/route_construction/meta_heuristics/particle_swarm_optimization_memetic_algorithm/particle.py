"""
Particle class for the Particle Swarm Optimization Memetic Algorithm (PSOMA).

Attributes:
    PSOMAParticle: Single-particle class holding position, velocity and personal best.

Example:
    >>> from logic.src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.particle import PSOMAParticle
    >>> particle = PSOMAParticle(routes=[[1, 2, 3]], profit=10.5)
    >>> print(particle.pbest_profit)
    10.5
"""

import copy
from typing import List


class PSOMAParticle:
    """
    A single PSO particle representing a routing solution.

    Attributes:
        routes: Current route set.
        profit: Objective value of current position.
        pbest_routes: Personal best route set.
        pbest_profit: Objective value of personal best.
    """

    def __init__(self, routes: List[List[int]], profit: float):
        """
        Initializes a PSO-MA particle.

        Args:
            routes: Initial routes for the particle.
            profit: Initial profit for the particle.
        """
        self.routes = routes
        self.profit = profit
        self.pbest_routes = copy.deepcopy(routes)
        self.pbest_profit = profit
