"""
Fast Iterative Localized Optimization (FILO) policy module.

This module provides the main entry point for the FILO metaheuristic,
incorporating Ruin & Recreate shaking and Local Search via Simulated Annealing.

References:
    Accorsi, L., & Vigo, D. "A fast and scalable heuristic for the solution
    of large-scale capacitated vehicle routing problems", 2021.
"""

import copy
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.fast_iterative_localized_optimization.params import FILOParams
from logic.src.policies.fast_iterative_localized_optimization.ruin_recreate import (
    RuinAndRecreate,
)
from logic.src.policies.other.local_search.local_search_filo import (
    FILOLocalSearch,
)
from logic.src.policies.other.operators.heuristics.greedy_initialization import (
    build_greedy_routes,
)


class FILOSolver:
    """
    Implementation of Fast Iterative Localized Optimization (FILO) for CVRP/VRPP.

    FILO maintains dynamic, localized parameters (gamma and omega) that restrict
    the search space during Local Search, leading to highly scalable iterations.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: FILOParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize the FILO solver.

        Args:
            dist_matrix: NxN distance matrix.
            wastes: Dictionary of node wastes.
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier per kg.
            C: Cost multiplier per km.
            params: Algorithm hyperparameters.
            mandatory_nodes: List of mandatory nodes.
        """
        self.d = dist_matrix
        self.waste = wastes
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params

        self.mandatory_nodes = mandatory_nodes or []
        self.mandatory_nodes_set = set(self.mandatory_nodes)

        self.n_nodes = len(dist_matrix) - 1
        self.all_customers = [n for n in self.waste.keys() if n != 0]

        # Use fixed numpy generator for reproducibility
        self.random = random.Random(self.params.seed)
        self.rng = np.random.default_rng(self.params.seed)

        # Initialize the Profit-Aware Ruin & Recreate operator
        self.ruin_recreate = RuinAndRecreate(
            dist_matrix=self.d,
            wastes=self.waste,
            capacity=self.Q,
            R=self.R,
            C=self.C,
            rng=self.rng,
            profit_aware_operators=self.params.profit_aware_operators,
            vrpp=self.params.vrpp,
        )

        # FILO-specific Local Search with active node localization support
        self.local_search = FILOLocalSearch(
            dist_matrix=self.d,
            waste=self.waste,
            capacity=self.Q,
            R=self.R,
            C=self.C,
            params=self.params,
        )

        # Node-specific gamma parameters (activation probability)
        self.node_gamma = [self.params.gamma_base] * (self.n_nodes + 1)

        # Node-specific omega parameters (shaking intensity L_it)
        omega_base = max(1, int(math.ceil(self.params.omega_base_multiplier * math.log(self.n_nodes + 1))))
        self.node_omega = [omega_base] * (self.n_nodes + 1)

        # Global omega intensity for current iteration (average of node-specific omegas)
        self.omega = 1.0

    def _evaluate_routes(self, routes: List[List[int]]) -> Tuple[float, float]:
        """Evaluate VRPP cost and profit."""
        total_cost = 0.0
        total_revenue = 0.0
        for route in routes:
            if not route:
                continue

            # Distance calculations
            prev = 0
            for node in route:
                total_cost += self.d[prev, node] * self.C
                total_revenue += self.waste.get(node, 0.0) * self.R
                prev = node
            total_cost += self.d[prev, 0] * self.C

        return total_cost, total_revenue - total_cost

    def _get_omega_nodes(self, current_routes: List[List[int]]) -> List[int]:
        r"""
        Extract spatial neighborhood nodes for ruin phase.

        Follows Accorsi & Vigo (2021):
        1. Select a 'center' node $i$ with probability proportional to $\gamma_i$.
        2. Identify the $L_{it}$ closest neighbors of $i$ (from node_omega[i]).
        3. Return this localized spatial neighborhood for destruction.

        Returns:
            List of node IDs forming the localized destruction neighborhood.
        """
        visited = []
        for r in current_routes:
            visited.extend(r)

        if not visited:
            return []

        # 1. Select center node i based on gamma_i
        # Nodes with high gamma (stagnant/unsuccessful) are more likely to be centers
        gammas = np.array([self.node_gamma[n] for n in visited], dtype=np.float64)
        probs = gammas / gammas.sum()
        center_node = self.rng.choice(visited, p=probs)

        # 2. Determine L_it (shaking intensity for this center node)
        l_it = self.node_omega[center_node]

        # 3. Get L_it closest neighbors of center_node that are currently visited
        distances = [(n, self.d[center_node, n]) for n in visited]
        distances.sort(key=lambda x: x[1])

        omega_nodes = [n for n, d in distances[:l_it]]
        return omega_nodes

    def _update_gamma(self, is_new_best: bool, accepted: bool, ruined: List[int]) -> None:
        """
        Update localized gamma (activation probability) parameters.

        Accorsi & Vigo (2021):
        - If new best: reset all gamma to gamma_base.
        - If NOT accepted: increase gamma for ruined nodes to intensify search there.
        - If accepted: reset gamma for ruined nodes.
        """
        if is_new_best:
            self.node_gamma = [self.params.gamma_base] * (self.n_nodes + 1)
            return

        if not accepted:
            for i in ruined:
                self.node_gamma[i] = min(1.0, self.node_gamma[i] + self.params.delta_gamma)
        else:
            for i in ruined:
                self.node_gamma[i] = self.params.gamma_base

    def _update_omega(self, is_new_best: bool, accepted: bool, ruined: List[int]) -> None:
        """
        Update localized omega (shaking intensity) parameters.

        Accorsi & Vigo (2021):
        - If new best: reset all omega to base.
        - If NOT accepted: increase L_it for ruined nodes (diversification).
        - If accepted: reset L_it.

        Also updates the global omega intensity for the next iteration.
        """
        omega_base = max(1, int(math.ceil(self.params.omega_base_multiplier * math.log(self.n_nodes + 1))))

        if is_new_best:
            self.node_omega = [omega_base] * (self.n_nodes + 1)
        elif not accepted:
            for i in ruined:
                # Increase intensity L_it up to a hard cap
                # Cap at 50 to prevent massive random restarts on large instances (Accorsi & Vigo 2021)
                self.node_omega[i] = min(50, self.node_omega[i] + 1)
        else:
            for i in ruined:
                self.node_omega[i] = omega_base

        # Update global omega intensity as normalized average of node-specific values
        # This scales the number of nodes removed in the ruin phase
        visited_nodes = [n for r in self.routes if r for n in r] if hasattr(self, "routes") else []
        if visited_nodes:
            # Normalize to ~1.0 baseline
            avg_omega = np.mean([self.node_omega[n] for n in visited_nodes])
            self.omega = avg_omega / omega_base  # type: ignore[assignment]
        else:
            self.omega = 1.0

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Execute the FILO heuristic.

        Returns:
            Tuple of (Best Routes, Best Profit, Best Cost)
        """
        start_time = time.process_time()

        # Step 1: Constructive Initialization (Profit Aware & Mandatory Respecting)
        current_routes = build_greedy_routes(
            dist_matrix=self.d,
            wastes=self.waste,
            capacity=self.Q,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
        )

        current_cost, current_profit = self._evaluate_routes(current_routes)

        best_routes = copy.deepcopy(current_routes)
        best_profit = current_profit
        best_cost = current_cost

        # Store routes for omega calculation
        self.routes = current_routes

        # Simulated Annealing Setup
        if current_cost > 0:
            self.sa_start_temp = current_cost / self.params.initial_temperature_factor
            self.sa_final_temp = current_cost / self.params.final_temperature_factor
        else:
            self.sa_start_temp = 100.0
            self.sa_final_temp = 1.0

        temperature = self.sa_start_temp

        for iteration in range(self.params.max_iterations):
            elapsed = time.process_time() - start_time
            if self.params.time_limit > 0 and elapsed > self.params.time_limit:
                break

            # --- STEP 1: GET OMEGA NODES (Spatial Neighborhood for Ruin) ---
            omega_nodes = self._get_omega_nodes(current_routes)

            # --- STEP 2: SHAKING (Ruin & Recreate with Omega Intensity) ---
            new_routes, num_ruined, ruined = self.ruin_recreate.apply(
                routes=current_routes,
                omega=omega_nodes,
                all_customers=self.all_customers,
                mandatory_nodes=self.mandatory_nodes,
                omega_intensity=self.omega,  # Pass global intensity parameter
            )

            # --- STEP 3: CALCULATE ACTIVE NODES (Gamma Localization) ---
            # Active set = Recently disrupted nodes + Historically stagnant nodes
            active_nodes = set(ruined)  # Start with recently ruined nodes

            # Add nodes with high gamma (stagnant nodes that need attention)
            for i in range(1, self.n_nodes):
                if self.node_gamma[i] > self.params.gamma_base:
                    active_nodes.add(i)

            # --- STEP 4: LOCAL SEARCH (Restricted to Active Neighborhood) ---
            ls_routes = self.local_search.optimize(new_routes, active_nodes=active_nodes)

            # --- EVALUATION ---
            ls_cost, ls_profit = self._evaluate_routes(ls_routes)
            delta_profit = ls_profit - current_profit

            # Simulated Annealing Move Acceptance
            accept = False
            if delta_profit > 1e-6:
                accept = True
            elif temperature > 0:
                # delta_profit is negative here, so math.exp(delta_profit / temperature) is <= 1
                prob = math.exp(delta_profit / temperature)
                if self.rng.random() < prob:
                    accept = True

            is_new_best = False
            if ls_profit > best_profit + 1e-6:
                best_routes = copy.deepcopy(ls_routes)
                best_profit = ls_profit
                best_cost = ls_cost
                is_new_best = True

            self._update_gamma(is_new_best, accept, ruined)
            self._update_omega(is_new_best, accept, ruined)

            if accept:
                current_routes = ls_routes
                current_profit = ls_profit
                current_cost = ls_cost
                self.routes = current_routes  # Update for omega calculation

            # Annealing Schedule
            if temperature > self.sa_final_temp:
                cooling_factor = (self.sa_final_temp / self.sa_start_temp) ** (1.0 / self.params.max_iterations)
                temperature *= cooling_factor

            # Visualization Support
            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                current_profit=current_profit,
                temperature=temperature,
                accepted=int(accept),
                score=3 if is_new_best else (1 if accept else 0),
            )

        return best_routes, best_profit, best_cost
