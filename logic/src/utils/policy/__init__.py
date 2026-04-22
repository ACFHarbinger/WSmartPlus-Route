"""
Helper files for local search, neighborhood search, crossover, and perturbation
operators, as well as heuristics and solution initialization schemas, for the VRPP.

Attributes:
    llh_pool: Low-Level Heuristic Pool for hyper-heuristics.
    neighborhood: Neighborhood search operators.
    routes: Route operators.
    wrappers: Wrappers for local search operators.

Example:
    >>> from logic.src.utils.policy import *
    >>> llh_pool.h1_greedy_move(problem, route, rng)
    >>> neighborhood.get_p_neighborhood(0, tour_nodes, dist_matrix, p=2)
    >>> routes.prune_unprofitable_routes(problem)
    >>> wrappers.initial_plan_greedy(problem)
"""
