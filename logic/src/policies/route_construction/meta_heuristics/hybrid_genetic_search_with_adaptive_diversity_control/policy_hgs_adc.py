"""
Hybrid Genetic Search with Adaptive Diversity Control (HGS-ADC) for MPVRP.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.pipeline.simulations.bins.prediction import ScenarioTree
from logic.src.policies.helpers.local_search.local_search_manager import LocalSearchManager
from logic.src.policies.helpers.operators.crossover_recombination.pattern_and_itinerary_crossover import (
    pattern_itinerary_crossover,
)
from logic.src.policies.helpers.operators.inter_route_local_search.inter_day_shift import inter_day_shift
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.registry import RouteConstructorRegistry

from .individual import Individual
from .initialization import generate_initial_individual
from .population import Population
from .split import compute_daily_loads, split_day


@RouteConstructorRegistry.register("hgs_adc")
class PolicyHGSADC(BaseMultiPeriodRoutingPolicy):
    """
    MPVRP Solver using Hybrid Genetic Search with Adaptive Diversity Control.
    Implements a generation loop with open unconstrained patterns and split algorithm.
    """

    def __init__(self, config: Any = None):
        super().__init__(config)

    def _run_multi_period_solver(
        self,
        tree: ScenarioTree,
        capacity: float,
        revenue: float,
        cost_unit: float,
        **kwargs: Any,
    ) -> Tuple[List[List[List[int]]], float, Dict[str, Any]]:
        T = tree.horizon
        base_wastes = tree.root.wastes.copy()
        N = len(base_wastes)

        daily_increments = np.zeros((T, N))
        previous_wastes = base_wastes

        for t in range(1, T + 1):
            scenarios = tree.get_scenarios_at_day(t)
            if scenarios:
                best_scenario = max(scenarios, key=lambda s: s.probability)
                current_wastes = best_scenario.wastes
                daily_increments[t - 1] = np.maximum(0, current_wastes - previous_wastes)
                previous_wastes = current_wastes

        dist_matrix = kwargs.get("distance_matrix")

        best_ind = self._construct_multi_period_routes(
            T, base_wastes, daily_increments, np.array(dist_matrix), capacity, **kwargs
        )

        if best_ind is None:
            return [[[0, 0]]] * T, 0.0, {}

        full_plan = []
        for t in range(T):
            routes = best_ind.routes[t]
            day_plan = []
            for r in routes:
                if len(r) > 0:
                    day_plan.append([0] + list(r) + [0])
            if not day_plan:
                day_plan = [[0, 0]]
            full_plan.append(day_plan)

        expected_profit = -best_ind.fit
        return full_plan, expected_profit, {"generations": getattr(self.config, "generations", 50)}

    def _construct_multi_period_routes(
        self,
        T: int,
        base_wastes: np.ndarray,
        daily_increments: np.ndarray,
        dist_matrix: np.ndarray,
        capacity: float,
        **kwargs: Any,
    ) -> Optional[Individual]:
        pop_size = getattr(self.config, "pop_size", 25)
        nb_close = getattr(self.config, "nb_close", 4)
        n_gen = getattr(self.config, "generations", 50)

        n_vehicles = 0
        if "config" in kwargs and isinstance(kwargs["config"], dict):
            n_vehicles = kwargs["config"].get("n_vehicles", 0)
        if getattr(self.config, "n_vehicles", None) is not None:
            n_vehicles = self.config.n_vehicles

        N = len(base_wastes)
        population = Population(pop_size, nb_close)

        for _ in range(pop_size * 2):
            ind = generate_initial_individual(N, T, base_wastes, daily_increments, dist_matrix, capacity)
            self._evaluate_individual(ind, base_wastes, daily_increments, dist_matrix, capacity, n_vehicles, T)
            population.add_individual(ind)

        population.trigger_survivor_selection(T)

        for _gen in range(n_gen):
            pool = population.feas + population.inf
            if len(pool) < 2:
                break

            p1 = random.choice(pool)
            p2 = random.choice(pool)
            while p2 == p1 and len(pool) > 1:
                p2 = random.choice(pool)

            child = pattern_itinerary_crossover(p1, p2, T, N)

            self._evaluate_individual(child, base_wastes, daily_increments, dist_matrix, capacity, n_vehicles, T)

            loads = compute_daily_loads(child.patterns, base_wastes, daily_increments, T)
            improved_routes = self.run_intra_day_local_search(child.routes, dist_matrix, capacity, loads)

            child.routes = improved_routes
            for t in range(T):
                gt = []
                for r in improved_routes[t]:
                    gt.extend(r)
                child.giant_tours[t] = np.array(gt, dtype=int)

            self._evaluate_individual(child, base_wastes, daily_increments, dist_matrix, capacity, n_vehicles, T)

            if random.random() < 0.2:
                inter_day_shift(child, base_wastes, daily_increments, dist_matrix, capacity, n_vehicles, T)

            population.add_individual(child)

            if len(population.feas) + len(population.inf) > pop_size * 2:
                population.trigger_survivor_selection(T)

        if population.feas:
            return sorted(population.feas, key=lambda x: x.fit)[0]
        elif population.inf:
            return sorted(population.inf, key=lambda x: x.fit)[0]

        return None

    def _evaluate_individual(
        self,
        ind: Individual,
        base_wastes: np.ndarray,
        daily_increments: np.ndarray,
        dist: np.ndarray,
        capacity: float,
        n_vehicles: int,
        T: int,
    ) -> None:
        loads = compute_daily_loads(ind.patterns, base_wastes, daily_increments, T)
        tot_cost = 0.0
        tot_viol = 0.0
        all_routes = []

        for t in range(T):
            r, c, v = split_day(ind.giant_tours[t], loads[t], dist, capacity, n_vehicles)
            all_routes.append(r)
            tot_cost += c
            tot_viol += v

        w_q = 100.0
        ind.cost = tot_cost
        ind.capacity_violations = tot_viol
        ind.fit = tot_cost + w_q * tot_viol
        ind.is_feasible = tot_viol == 0.0
        ind.routes = all_routes

    @staticmethod
    def run_intra_day_local_search(
        routes: List[List[List[int]]],
        dist: np.ndarray,
        capacity: float,
        loads: np.ndarray,
    ) -> List[List[List[int]]]:
        """
        Intra-day Local Search (Education).
        Applies unified local search operators (2-opt, SWAP*)
        to the given decoded routes on a day-by-day basis.
        """
        improved_routes: List[List[List[int]]] = []
        T = len(routes)

        for t in range(T):
            t_routes = routes[t]
            if not t_routes:
                improved_routes.append([])
                continue

            load_array = loads[t]
            # Build waste dict for LocalSearchManager
            wastes = {i: float(load_array[i]) for i in range(len(load_array))}

            ls = LocalSearchManager(
                dist_matrix=dist, wastes=wastes, capacity=capacity, R=0.0, C=1.0, improvement_threshold=1e-4
            )
            ls.penalty_capacity = 100.0  # type: ignore
            ls.set_routes(t_routes)

            improved = True
            while improved:
                improved = False
                if ls.two_opt_intra():
                    improved = True
                if ls.swap_star():
                    improved = True

            improved_routes.append(ls.get_routes())

        return improved_routes
