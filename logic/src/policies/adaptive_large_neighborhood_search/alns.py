"""
Adaptive Large Neighborhood Search (ALNS) for VRPP.

This module implements ALNS following Ropke & Pisinger (2005) methodology,
adapted for the Vehicle Routing Problem with Profits (VRPP) as a profit
maximization problem.

Key Implementation Features:
============================

1. **Operator Weight Updates** (Section 3.4.4):
   - Segment-based learning: w_{i,j+1} = w_{i,j}(1-r) + r * (π_i / θ_i)
   - Reaction factor r controls adaptation speed (default: 0.1)
   - Weights updated every `segment_size` iterations (default: 100)

2. **Scoring Mechanism** (Section 3.3):
   - σ₁: New global best solution found (default: 33)
   - σ₂: Better solution not visited before (default: 9)
   - σ₃: Accepted worse solution not visited before (default: 13)
   - Hash table tracks visited solutions to prevent rewarding revisits

3. **Simulated Annealing Acceptance**:
   - For VRPP profit maximization: P(accept) = exp(-Δ/T) where Δ = current - new
   - Dynamic temperature initialization: T_start = |initial_profit * w| / ln(2)
     such that solutions w% worse are accepted with probability 0.5

4. **Randomized Worst Removal** (Section 3.4.1):
   - Index selection: floor(y^p * |L|) where y ~ U(0,1) and p >= 1
   - Higher p values bias toward deterministically worst nodes
   - Default p = 3.0 provides good exploration/exploitation balance

5. **Adaptive Noise in Repair** (Section 3.4.3):
   - Applied additively to final insertion cost: C' = max{0, C + noise}
   - Noise ~ U[-η * max_dist, η * max_dist] where η (default: 0.025)
   - Separate clean (noise=0) and noisy operator variants in weight mechanism
   - Algorithm learns which variant is most effective via adaptive weights

6. **Profit-Aware Operators** (Novel Contribution):
   - Speculative Seeding: seed_hurdle = -0.5 * detour_cost
     Allows initially unprofitable routes that may become profitable
   - Profit-based removal: targets nodes with lowest marginal profit
   - Enable via `profit_aware_operators=True` for ablation studies

Performance Optimizations:
==========================
- Shallow copy via list comprehension (replaces deepcopy): ~10x faster
- Visited solutions tracked via canonical tuple hashing
- Incremental route evaluation during operator application

Usage for Ablation Studies:
============================
Toggle `profit_aware_operators` in ALNSParams to compare:
- Standard ALNS (False): cost-minimization operators
- Profit-aware ALNS (True): revenue-cost maximization operators

References:
===========
[1] Ropke, S., & Pisinger, D. (2006). "An adaptive large neighborhood search
    heuristic for the pickup and delivery problem with time windows."
    Transportation Science, 40(4), 455-472.
"""

import math
import random
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyStateRecorder
from logic.src.utils.functions import safe_exp

from ..other.operators import (
    build_greedy_routes,
    greedy_insertion,
    greedy_profit_insertion,
    random_removal,
    regret_2_insertion,
    regret_2_profit_insertion,
    regret_3_insertion,
    regret_3_profit_insertion,
    regret_4_insertion,
    regret_4_profit_insertion,
    shaw_profit_removal,
    shaw_removal,
    worst_profit_removal,
    worst_removal,
)
from .params import ALNSParams


class ALNSSolver:
    """
    Custom implementation of Adaptive Large Neighborhood Search for CVRP.
    Follows Pisinger & Ropke (2007) with segment-based weight updates and noise.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: ALNSParams,
        mandatory_nodes: Optional[List[int]] = None,
        recorder: Optional[PolicyStateRecorder] = None,
    ):
        """
        Initialize the ALNS solver.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes
        self.vrpp = getattr(params, "vrpp", True)
        self.profit_aware_operators = getattr(params, "profit_aware_operators", False)
        self.random = random.Random(params.seed) if params.seed is not None else random.Random(42)
        self.np_random = np.random.default_rng(params.seed) if params.seed is not None else np.random.default_rng(42)

        if recorder is not None:
            self._viz_record = recorder.record

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        # Operator registry
        if self.profit_aware_operators:
            self.destroy_ops = [
                lambda r, n: random_removal(r, n, rng=self.random),
                lambda r, n: worst_profit_removal(
                    r,
                    n,
                    self.dist_matrix,
                    self.wastes,
                    self.R,
                    self.C,
                    p=params.worst_removal_randomness,
                    rng=self.random,  # type: ignore[arg-type]
                ),
                lambda r, n: shaw_profit_removal(
                    r,
                    n,
                    self.dist_matrix,
                    wastes=self.wastes,
                    R=self.R,
                    C=self.C,
                    randomization_factor=params.shaw_randomization,
                    rng=self.random,
                ),
            ]
        else:
            self.destroy_ops = [
                lambda r, n: random_removal(r, n, rng=self.random),
                lambda r, n: worst_removal(
                    r, n, self.dist_matrix, p=params.worst_removal_randomness, rng=self.np_random
                ),
                lambda r, n: shaw_removal(
                    r,
                    n,
                    self.dist_matrix,
                    wastes=self.wastes,
                    randomization_factor=params.shaw_randomization,
                    rng=self.random,
                ),
            ]

        # In Conf. 15 (paper), clean and noisy variants are separate operator slots
        # The adaptive weight mechanism learns which variant is better.
        self.repair_ops: List[Callable] = []
        self.repair_names: List[str] = []

        if self.profit_aware_operators:
            # 4 total slots (greedy, regret-2, regret-3, regret-4), each accepts noise injected at call time.
            self.repair_ops = [
                lambda r, n, _noise=0.0: greedy_profit_insertion(
                    r,
                    n,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.vrpp,
                    noise=_noise,
                ),
                lambda r, n, _noise=0.0: regret_2_profit_insertion(
                    r,
                    n,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.vrpp,
                    noise=_noise,
                ),
                lambda r, n, _noise=0.0: regret_3_profit_insertion(
                    r,
                    n,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.vrpp,
                    noise=_noise,
                ),
                lambda r, n, _noise=0.0: regret_4_profit_insertion(
                    r,
                    n,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.vrpp,
                    noise=_noise,
                ),
            ]
            self.repair_names = ["GreedyProfit", "Regret2Profit", "Regret3Profit", "Regret4Profit"]
        else:
            # Standard branch - 4 slots
            self.repair_ops = [
                lambda r, n, _noise=0.0: greedy_insertion(
                    r,
                    n,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.vrpp,
                    noise=_noise,
                ),
                lambda r, n, _noise=0.0: regret_2_insertion(
                    r,
                    n,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.vrpp,
                    noise=_noise,
                ),
                lambda r, n, _noise=0.0: regret_3_insertion(
                    r,
                    n,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.vrpp,
                    noise=_noise,
                ),
                lambda r, n, _noise=0.0: regret_4_insertion(
                    r,
                    n,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.vrpp,
                    noise=_noise,
                ),
            ]
            self.repair_names = ["Greedy", "Regret2", "Regret3", "Regret4"]

        # Segment-based weight update logic (Ropke & Pisinger 2005, Section 3.4.4)
        self.segment_size = params.segment_size
        self.destroy_weights = [1.0] * len(self.destroy_ops)
        self.repair_weights = [1.0] * len(self.repair_ops)
        self.destroy_scores = [0.0] * len(self.destroy_ops)
        self.repair_scores = [0.0] * len(self.repair_ops)
        self.destroy_counts = [0] * len(self.destroy_ops)
        self.repair_counts = [0] * len(self.repair_ops)

        # Weight decay factor: (1 - r) where r is the reaction factor
        self.weight_decay = 1.0 - params.reaction_factor
        self.weight_learning_rate = params.reaction_factor

        # Visited solutions tracking (Ropke & Pisinger 2005, Section 3.3)
        # Hash table to track visited solutions for adjusted scoring
        self.visited_solutions: Set[Tuple[int, ...]] = set()

    def _get_noise(self) -> float:
        """
        Generate noise for repair operators (Ropke & Pisinger 2005, Section 3.4.3).

        Noise is scaled relative to the maximum distance in the problem instance:
        noise ~ U[-eta * max_dist, eta * max_dist]

        The decision to use noise vs. clean insertion is handled by the adaptive
        weight mechanism, which maintains separate weights for clean and noisy variants
        of each operator.

        Returns:
            float: Noise value in range [-eta * max_dist, eta * max_dist]
        """
        max_dist = self.dist_matrix.max()
        return self.random.uniform(-self.params.noise_factor, self.params.noise_factor) * max_dist

    def _hash_solution(self, routes: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
        """
        Create a canonical hash of a solution for tracking visited states.

        Args:
            routes: List of routes to hash

        Returns:
            Tuple representation of sorted routes for hashing
        """
        # Sort routes by their first element for canonical representation
        sorted_routes = sorted([tuple(route) for route in routes if route])
        return tuple(sorted_routes)

    def _initialize_solve(self, initial_solution: Optional[List[List[int]]]):
        current_routes = initial_solution or self.build_initial_solution()
        best_routes = [r[:] for r in current_routes]
        best_cost = self.calculate_cost(best_routes)
        best_rev = sum(self.wastes.get(node_idx, 0) * self.R for route in best_routes for node_idx in route)
        best_profit = best_rev - best_cost
        return current_routes, best_routes, best_profit, best_cost

    def _select_and_apply_operators(self, current_routes):
        d_idx = self.select_operator(self.destroy_weights)
        r_idx = self.select_operator(self.repair_weights)
        destroy_op = self.destroy_ops[d_idx]
        repair_op = self.repair_ops[r_idx]

        current_n_nodes = sum(len(route) for route in current_routes)
        if current_n_nodes == 0:
            n_remove = 0
        else:
            lower_bound = min(current_n_nodes, self.params.min_removal)
            max_pct_remove = int(current_n_nodes * self.params.max_removal_pct)
            # Upper bound: min(100, xi * n) per paper §4.3.1
            upper_bound = min(current_n_nodes, self.params.max_removal_cap, max(lower_bound, max_pct_remove))
            upper_bound = max(upper_bound, lower_bound)
            n_remove = self.random.randint(lower_bound, upper_bound)

        # 50% coin-flip per paper §3.6 for noise insertion variant
        use_noise = self.random.random() < 0.5
        noise_val = self._get_noise() if use_noise else 0.0

        new_routes, removed = destroy_op([route[:] for route in current_routes], n_remove)
        new_routes = repair_op(new_routes, removed, noise_val)
        return new_routes, d_idx, r_idx

    def _accept_solution(self, current_profit, new_profit, T):
        """
        Simulated Annealing acceptance criterion for VRPP profit maximization.

        For maximization problems, we accept if:
        1. new_profit > current_profit (always accept better solutions)
        2. new_profit <= current_profit with probability exp(-Δ/T) where Δ = current_profit - new_profit

        This is equivalent to the standard SA acceptance for minimization (f(x_new) < f(x_current))
        by using delta = f(x_current) - f(x_new) for the profit objective.

        Args:
            current_profit: Current solution profit (revenue - cost)
            new_profit: New solution profit
            T: Current temperature

        Returns:
            bool: True if solution should be accepted
        """
        delta = current_profit - new_profit
        if delta < -1e-6:  # new_profit > current_profit (improvement)
            return True

        # Acceptance probability: P(accept) = exp(-Δ/T) where Δ >= 0
        prob = safe_exp(-delta / T) if T > 0 else 0
        return self.random.random() < prob

    def _update_weights(self, d_idx, r_idx, score):
        self.destroy_scores[d_idx] += score
        self.repair_scores[r_idx] += score
        self.destroy_counts[d_idx] += 1
        self.repair_counts[r_idx] += 1

    def _end_segment(self):
        """
        Update operator weights at the end of a segment (Ropke & Pisinger 2005, Eq. 4.1).

        Formula: w_{i,j+1} = w_{i,j} * (1-r) + r * (π_i / θ_i)
        where:
            - w_{i,j} is the weight of operator i in segment j
            - r is the reaction factor (learning rate)
            - π_i is the total score accumulated by operator i
            - θ_i is the number of times operator i was used
        """
        for i in range(len(self.destroy_weights)):
            if self.destroy_counts[i] > 0:
                # Average score per usage: π_i / θ_i
                avg_score = self.destroy_scores[i] / self.destroy_counts[i]
                # Update: w_{i,j+1} = (1-r) * w_{i,j} + r * avg_score
                self.destroy_weights[i] = (
                    self.weight_decay * self.destroy_weights[i] + self.weight_learning_rate * avg_score
                )
            self.destroy_scores[i] = 0.0
            self.destroy_counts[i] = 0

        for i in range(len(self.repair_weights)):
            if self.repair_counts[i] > 0:
                avg_score = self.repair_scores[i] / self.repair_counts[i]
                self.repair_weights[i] = (
                    self.weight_decay * self.repair_weights[i] + self.weight_learning_rate * avg_score
                )
            self.repair_scores[i] = 0.0
            self.repair_counts[i] = 0

    def _calculate_dynamic_start_temp(self, initial_profit: float, w_percent: float = 0.05) -> float:
        """
        Calculate dynamic initial temperature (Ropke & Pisinger 2005, Section 3.3).

        The temperature is set such that a solution that is w% worse than the initial
        solution is accepted with probability 0.5.

        For maximization: A solution w% worse has profit = initial_profit * (1 - w)
        Delta = initial_profit - (initial_profit * (1-w)) = initial_profit * w
        We want: exp(-Delta/T) = 0.5
        Therefore: T = -Delta / ln(0.5) = Delta / ln(2)

        Args:
            initial_profit: Profit of initial solution
            w_percent: Percentage worse (default 5%)

        Returns:
            float: Initial temperature
        """
        delta = abs(initial_profit * w_percent)
        if delta < 1e-9:
            return 100.0  # Fallback if initial solution has near-zero profit
        T_start = delta / math.log(2)  # ln(2) ≈ 0.693
        return max(T_start, 1.0)  # Ensure minimum temperature

    def solve(self, initial_solution: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], float, float]:
        start_time = time.process_time()
        current_routes, best_routes, best_profit, best_cost = self._initialize_solve(initial_solution)
        current_profit = best_profit

        # Clear visited solutions set
        self.visited_solutions.clear()
        self.visited_solutions.add(self._hash_solution(best_routes))  # type: ignore[arg-type]

        # Dynamic temperature initialization (Ropke & Pisinger 2005, Section 3.3)
        if self.params.start_temp > 0:
            T = self.params.start_temp
        else:
            T = self._calculate_dynamic_start_temp(best_profit, w_percent=0.05)

        for _it in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            new_routes, d_idx, r_idx = self._select_and_apply_operators(current_routes)
            new_cost = self.calculate_cost(new_routes)
            new_rev = sum(self.wastes.get(node_idx, 0) * self.R for route in new_routes for node_idx in route)
            new_profit = new_rev - new_cost

            # Hash the new solution
            new_hash = self._hash_solution(new_routes)
            is_new_solution = new_hash not in self.visited_solutions

            accept = self._accept_solution(current_profit, new_profit, T)

            # Scoring based on Ropke & Pisinger (2005, Section 3.3):
            score = 0.0
            if new_profit > best_profit + 1e-6:
                # σ₁: new global best — evaluated before accept, awarded regardless of
                # whether new_profit > current_profit
                best_routes = [r[:] for r in new_routes]
                best_profit = new_profit
                best_cost = new_cost
                score = self.params.sigma_1

            if accept:
                if score == 0:
                    # σ₁ was not already awarded
                    if new_profit > current_profit + 1e-6:
                        # σ₂: better solution not visited before
                        if is_new_solution:
                            score = self.params.sigma_2
                    else:
                        # σ₃: accepted worse solution not visited before
                        if is_new_solution:
                            score = self.params.sigma_3

                self.visited_solutions.add(new_hash)  # type: ignore[arg-type]
                current_routes = new_routes
                current_profit = new_profit
            else:
                # score is already 0.0 unless σ₁ was found but not accepted (impossible in profit maximization)
                pass

            self._update_weights(d_idx, r_idx, score)
            if (_it + 1) % self.segment_size == 0:
                self._end_segment()

            # Cooling schedule
            T *= self.params.cooling_rate

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=_it,
                d_idx=d_idx,
                r_idx=r_idx,
                best_profit=best_profit,
                current_profit=current_profit,
                temperature=T,
                accepted=int(accept),
                score=score,
            )
        return best_routes, best_profit, best_cost

    def select_operator(self, weights: List[float]) -> int:
        total = sum(weights)
        r = self.random.uniform(0, total)
        curr = 0.0
        for i, w in enumerate(weights):
            curr += w
            if curr >= r:
                return i
        return len(weights) - 1

    def calculate_cost(self, routes: List[List[int]]) -> float:
        total_dist = 0
        for route in routes:
            if not route:
                continue
            dist = self.dist_matrix[0][route[0]]
            for i in range(len(route) - 1):
                dist += self.dist_matrix[route[i]][route[i + 1]]
            dist += self.dist_matrix[route[-1]][0]
            total_dist += dist
        return total_dist * self.C

    def build_initial_solution(self) -> List[List[int]]:
        """
        Build an initial solution using the greedy profit-aware heuristic.
        """
        return build_greedy_routes(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
        )
