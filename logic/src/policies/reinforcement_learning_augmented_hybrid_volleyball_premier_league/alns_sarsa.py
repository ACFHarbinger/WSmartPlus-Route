"""
Enhanced Adaptive Large Neighborhood Search (ALNS) with SARSA Reinforcement Learning.

This module implements an advanced ALNS solver that uses SARSA (State-Action-Reward-State-Action)
reinforcement learning to intelligently select destroy/repair operators. It includes all available
operators from the codebase:

Destroy Operators:
    - Random removal
    - Worst removal
    - Cluster removal
    - Shaw removal (relatedness-based)
    - String removal (contiguous segments)

Repair Operators:
    - Greedy insertion
    - Regret-2 insertion
    - Regret-k insertion (extended)
    - Greedy blink insertion (randomized)

Perturbation Operators:
    - Route shuffling
    - Node sequence reversal
    - Random restart from strong perturbation

Unstringing Operators (SISR - Slack Induction by String Removal):
    - Type I: Single string removal
    - Type II: Multiple string removal
    - Type III: Propagating string removal
    - Type IV: Clustered string removal

Reference:
    Sutton & Barto, "Reinforcement Learning: An Introduction", 2nd Ed., 2018.
    Pisinger & Ropke, "A general heuristic for vehicle routing problems", 2007.
"""

import copy
import math
import random
import time
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy import (
    cluster_removal as cluster_removal_op,
)
from ..operators.destroy import (
    random_removal as random_removal_op,
)
from ..operators.destroy import (
    shaw_removal as shaw_removal_op,
)
from ..operators.destroy import (
    string_removal as string_removal_op,
)
from ..operators.destroy import (
    worst_removal as worst_removal_op,
)
from ..operators.perturbation import kick as kick_op
from ..operators.perturbation import perturb as perturb_op
from ..operators.repair import (
    greedy_insertion as greedy_insertion_op,
)
from ..operators.repair import (
    greedy_insertion_with_blinks as greedy_insertion_with_blinks_op,
)
from ..operators.repair import (
    regret_2_insertion as regret_2_insertion_op,
)
from ..operators.repair import (
    regret_k_insertion as regret_k_insertion_op,
)
from ..operators.unstringing import (
    apply_type_i_unstringing as type_i_removal_op,
)
from ..operators.unstringing import (
    apply_type_ii_unstringing as type_ii_removal_op,
)
from ..operators.unstringing import (
    apply_type_iii_unstringing as type_iii_removal_op,
)
from ..operators.unstringing import (
    apply_type_iv_unstringing as type_iv_removal_op,
)
from .params import ALNSParams, RLAHVPLParams


class SARSAOperatorSelector:
    """
    SARSA-based operator selection for ALNS.

    Uses State-Action-Reward-State-Action temporal difference learning
    to learn optimal destroy-repair operator combinations.
    """

    def __init__(
        self,
        n_destroy_ops: int,
        n_repair_ops: int,
        alpha: float = 0.1,  # Learning rate
        gamma: float = 0.95,  # Discount factor
        epsilon: float = 0.15,  # Exploration rate
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        scores_size: int = 50,
        qtable_size_rate: float = 0.5,
        progress_thresholds: List[float] = None,
        stagnation_thresholds: List[int] = None,
        diversity_thresholds: List[float] = None,
    ):
        """Initialize SARSA Q-learning for operator selection."""
        if progress_thresholds is None:
            progress_thresholds = [0.33, 0.67]
        if stagnation_thresholds is None:
            stagnation_thresholds = [10, 30]
        if diversity_thresholds is None:
            diversity_thresholds = [0.3, 0.7]
        self.n_destroy = n_destroy_ops
        self.n_repair = n_repair_ops
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.progress_thresholds = progress_thresholds
        self.stagnation_thresholds = stagnation_thresholds
        self.diversity_thresholds = diversity_thresholds
        # Q-table: Q[state][destroy_op][repair_op] = expected reward
        # State discretization: (iteration_phase, stagnation_level, diversity_level)
        self.q_table: Dict[Tuple[int, int, int], np.ndarray] = defaultdict(
            lambda: np.ones((n_destroy_ops, n_repair_ops)) * qtable_size_rate
        )

        # Performance tracking
        self.op_pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        self.op_pair_scores: Dict[Tuple[int, int], Deque[float]] = defaultdict(lambda: deque(maxlen=scores_size))

        # SARSA state tracking
        self.current_state: Optional[Tuple[int, int, int]] = None
        self.current_action: Optional[Tuple[int, int]] = None

    def discretize_state(
        self, iteration: int, max_iterations: int, stagnation_count: int, diversity: float
    ) -> Tuple[int, int, int]:
        """
        Discretize continuous state into bins for Q-table.

        Args:
            iteration: Current iteration number.
            max_iterations: Maximum iterations.
            stagnation_count: Number of iterations without improvement.
            diversity: Population diversity measure (0-1).

        Returns:
            Tuple of (phase, stagnation_level, diversity_level).
        """
        # Phase: early (0), mid (1), late (2)
        progress = iteration / max(max_iterations, 1)
        phase = 0 if progress < self.progress_thresholds[0] else (1 if progress < self.progress_thresholds[1] else 2)

        # Stagnation: low (0), medium (1), high (2)
        stag_level = (
            0
            if stagnation_count < self.stagnation_thresholds[0]
            else (1 if stagnation_count < self.stagnation_thresholds[1] else 2)
        )

        # Diversity: low (0), medium (1), high (2)
        div_level = (
            0 if diversity < self.diversity_thresholds[0] else (1 if diversity < self.diversity_thresholds[1] else 2)
        )

        return (phase, stag_level, div_level)

    def select_operators(self, state: Tuple[int, int, int], rng: random.Random) -> Tuple[int, int]:
        """
        Select destroy/repair operator pair using ε-greedy SARSA policy.

        Args:
            state: Discretized state tuple.
            rng: Random number generator.

        Returns:
            Tuple of (destroy_idx, repair_idx).
        """
        # ε-greedy exploration
        if rng.random() < self.epsilon:
            # Explore: random action
            d_idx = rng.randint(0, self.n_destroy - 1)
            r_idx = rng.randint(0, self.n_repair - 1)
        else:
            # Exploit: best action from Q-table
            q_values = self.q_table[state]
            best_value = np.max(q_values)

            # Find all actions with best value (for tie-breaking)
            best_actions = np.argwhere(q_values == best_value)
            if len(best_actions) > 0:
                choice = rng.choice(best_actions)
                d_idx, r_idx = int(choice[0]), int(choice[1])
            else:
                d_idx = rng.randint(0, self.n_destroy - 1)
                r_idx = rng.randint(0, self.n_repair - 1)

        return d_idx, r_idx

    def update_q_value(
        self,
        state: Tuple[int, int, int],
        action: Tuple[int, int],
        reward: float,
        next_state: Tuple[int, int, int],
        next_action: Tuple[int, int],
    ):
        """
        Update Q-value using SARSA temporal difference update.

        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        """
        d_idx, r_idx = action
        next_d_idx, next_r_idx = next_action

        current_q = self.q_table[state][d_idx, r_idx]
        next_q = self.q_table[next_state][next_d_idx, next_r_idx]

        # SARSA update
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        new_q = current_q + self.alpha * td_error

        self.q_table[state][d_idx, r_idx] = new_q

        # Track performance
        self.op_pair_counts[action] += 1
        self.op_pair_scores[action].append(reward)

    def decay_epsilon(self):
        """Decay exploration rate over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_statistics(self) -> Dict:
        """Get operator selection statistics for visualization."""
        stats = {
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
            "total_actions": sum(self.op_pair_counts.values()),
            "best_operators": [],
        }

        # Find best operator pairs
        avg_scores = {}
        for action, scores in self.op_pair_scores.items():
            if scores:
                avg_scores[action] = np.mean(scores)

        if avg_scores:
            sorted_pairs = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            stats["best_operators"] = sorted_pairs[:5]

        return stats


class ALNSPerturbationContext:
    """Context object required by perturbation operators."""

    def __init__(self, routes: List[List[int]], dist_matrix: np.ndarray, wastes: Dict[int, float], capacity: float):
        self.routes = copy.deepcopy(routes)
        self.dist_matrix = dist_matrix
        self.d = dist_matrix
        self.wastes = wastes
        self.Q = capacity
        self._build_structures()

    def _build_structures(self):
        self.node_map = {}
        for r_idx, route in enumerate(self.routes):
            for pos, node in enumerate(route):
                self.node_map[node] = (r_idx, pos)

    def _update_map(self, changed_routes: set):
        self._build_structures()

    def _get_load_cached(self, ri: int) -> float:
        return sum(self.wastes.get(n, 0) for n in self.routes[ri])


class ALNSSARSASolver(PolicyVizMixin):
    """
    Enhanced ALNS solver with SARSA reinforcement learning for operator selection.

    Includes all destroy, repair, perturbation, and unstringing operators.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: ALNSParams,
        rl_params: RLAHVPLParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
        evaluator=None,
    ):
        """Initialize enhanced ALNS-SARSA solver."""
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.evaluator = evaluator
        self.epsilon_decay_step = rl_params.sarsa_epsilon_decay_step
        self.mandatory_nodes = mandatory_nodes or []
        self.improvement_thresholds = rl_params.sarsa_improvement_thresholds
        self.random = random.Random(seed) if seed is not None else random.Random()

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        # Initialize all operator pools
        self._init_destroy_operators()
        self._init_repair_operators()
        self._init_unstring_operators()
        self._init_perturbation_operators()

        # Add unstring operators to destroy options
        self.destroy_ops.extend(self.unstring_ops)
        self.destroy_names.extend(self.unstring_names)

        # Initialize SARSA selector
        self.sarsa = SARSAOperatorSelector(
            n_destroy_ops=len(self.destroy_ops),
            n_repair_ops=len(self.repair_ops),
            alpha=rl_params.sarsa_alpha,
            gamma=rl_params.sarsa_gamma,
            epsilon=rl_params.sarsa_epsilon,
            epsilon_decay=rl_params.sarsa_epsilon_decay,
            epsilon_min=rl_params.sarsa_epsilon_min,
            scores_size=rl_params.sarsa_scores_size,
            qtable_size_rate=rl_params.sarsa_qtable_size_rate,
            progress_thresholds=rl_params.sarsa_operator_progress_thresholds,
            stagnation_thresholds=rl_params.sarsa_operator_stagnation_thresholds,
            diversity_thresholds=rl_params.sarsa_operator_diversity_thresholds,
        )

        # Tracking
        self.stagnation_count = 0
        self.diversity_history: Deque[float] = deque(maxlen=rl_params.sarsa_diversity_size)

    def _init_destroy_operators(self):
        """Initialize all destroy operators."""
        self.destroy_ops = [
            # Standard operators
            self._destroy_random,
            self._destroy_worst,
            self._destroy_cluster,
            # Advanced operators
            self._destroy_shaw,
            self._destroy_string,
        ]

        self.destroy_names = [
            "Random",
            "Worst",
            "Cluster",
            "Shaw",
            "String",
        ]

    def _init_repair_operators(self):
        """Initialize all repair operators."""
        self.repair_ops = [
            # Standard operators
            self._repair_greedy,
            self._repair_regret2,
            # Advanced operators
            lambda routes, removed: self._repair_regretk(routes, removed, k=3),
            lambda routes, removed: self._repair_regretk(routes, removed, k=4),
            self._repair_greedy_blink,
        ]

        self.repair_names = ["Greedy", "Regret-2", "Regret-3", "Regret-4", "Greedy-Blink"]

    def _init_unstring_operators(self):
        """Initialize all unstring operators."""
        self.unstring_ops = [
            self._unstring_type_i,
            self._unstring_type_ii,
            self._unstring_type_iii,
            self._unstring_type_iv,
        ]

        self.unstring_names = ["Type-I", "Type-II", "Type-III", "Type-IV"]

    def _init_perturbation_operators(self):
        """Initialize all perturbation operators."""
        self.perturbation_ops = [
            self._perturb_kick,
            self._perturb_random,
        ]

        self.perturbation_names = ["Kick", "Random"]

    # ===== Destroy Operators =====

    def _destroy_random(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Random removal."""
        return random_removal_op(routes, n, rng=self.random)

    def _destroy_worst(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Worst removal (highest cost nodes)."""
        return worst_removal_op(routes, n, self.dist_matrix)

    def _destroy_cluster(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Cluster removal (spatially related nodes)."""
        return cluster_removal_op(routes, n, self.dist_matrix, self.nodes, rng=self.random)

    def _destroy_shaw(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Shaw removal (multi-criteria relatedness)."""
        return shaw_removal_op(
            routes,
            n,
            self.dist_matrix,
            self.nodes,
            waste_dict=self.wastes,
            rng=self.random,
        )

    def _destroy_string(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """String removal (contiguous segments)."""
        return string_removal_op(routes, n, self.dist_matrix, rng=self.random)

    # ===== Repair Operators =====

    def _repair_greedy(self, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Greedy insertion."""
        return greedy_insertion_op(
            routes,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
            cost_unit=self.C,
        )

    def _repair_regret2(self, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Regret-2 insertion."""
        return regret_2_insertion_op(
            routes,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
            cost_unit=self.C,
        )

    def _repair_regretk(self, routes: List[List[int]], removed: List[int], k: int) -> List[List[int]]:
        """Regret-k insertion (extended from regret-2)."""
        return regret_k_insertion_op(
            routes,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
            cost_unit=self.C,
            k=k,
        )

    def _repair_greedy_blink(
        self, routes: List[List[int]], removed: List[int], blink_rate: float = 0.1
    ) -> List[List[int]]:
        """Greedy insertion with randomized blinks."""
        return greedy_insertion_with_blinks_op(
            routes,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            blink_rate=blink_rate,
            rng=self.random,
        )

    # ===== Unstringing Operators =====

    def _unstring_wrapper(self, routes: List[List[int]], n: int, op_type: int) -> Tuple[List[List[int]], List[int]]:
        """Wrapper to apply unstringing moves n times as a destroy operator."""
        removed = []
        for _ in range(n):
            valid_indices = [idx for idx, r in enumerate(routes) if len(r) > 4]
            if not valid_indices:
                break
            r_idx = self.random.choice(valid_indices)
            route = routes[r_idx]

            i = self.random.randint(1, len(route) - 2)
            node_to_remove = route[i]

            valid_targets = [idx for idx in range(1, len(route) - 1) if idx not in (i - 1, i, i + 1)]

            try:
                if op_type == 1 and len(valid_targets) >= 2:
                    j, k = self.random.sample(valid_targets, 2)
                    new_route = type_i_removal_op(route, i, j, k)
                elif op_type == 2 and len(valid_targets) >= 2:
                    j, k = sorted(self.random.sample(valid_targets, 2))
                    new_route = type_ii_removal_op(route, i, j, k)
                elif op_type == 3 and len(valid_targets) >= 3:
                    j, k, l = self.random.sample(valid_targets, 3)
                    new_route = type_iii_removal_op(route, i, j, k, l)
                elif op_type == 4 and len(valid_targets) >= 3:
                    j, k, l = self.random.sample(valid_targets, 3)
                    new_route = type_iv_removal_op(route, i, j, k, l)
                else:
                    new_route = route
            except Exception:
                new_route = route

            if len(new_route) < len(route):
                routes[r_idx] = new_route
                removed.append(node_to_remove)

        return routes, removed

    def _unstring_type_i(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Type-I unstringing."""
        return self._unstring_wrapper(routes, n, 1)

    def _unstring_type_ii(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Type-II unstringing."""
        return self._unstring_wrapper(routes, n, 2)

    def _unstring_type_iii(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Type-III unstringing."""
        return self._unstring_wrapper(routes, n, 3)

    def _unstring_type_iv(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Type-IV unstringing."""
        return self._unstring_wrapper(routes, n, 4)

    # ===== Perturbation Operators =====

    def _perturb_kick(self, routes: List[List[int]]) -> List[List[int]]:
        """Perturbation operator: removes random nodes and reinserts them greedily."""
        ctx = ALNSPerturbationContext(routes, self.dist_matrix, self.wastes, self.capacity)
        kick_op(ctx, destroy_ratio=0.2, rng=self.random)
        return ctx.routes

    def _perturb_random(self, routes: List[List[int]]) -> List[List[int]]:
        """Perturbation operator: performs random swaps to escape local optima."""
        ctx = ALNSPerturbationContext(routes, self.dist_matrix, self.wastes, self.capacity)
        perturb_op(ctx, k=self.params.perturb_k if hasattr(self.params, "perturb_k") else 3, rng=self.random)
        return ctx.routes

    # ===== Main Solve Loop =====

    def solve(self, initial_solution: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], float, float]:  # noqa: C901
        """
        Run enhanced ALNS with SARSA operator selection.

        Returns:
            Tuple of (best_routes, profit, cost).
        """
        # Initialize solution
        current_routes, best_routes, best_profit, best_cost = self._initialize_solve(initial_solution)
        current_profit = best_profit

        current_eval_profit = self.evaluator(current_routes) if self.evaluator else current_profit

        # Simulated annealing temperature
        T = self.params.start_temp
        start_time = time.process_time()

        # SARSA: Initialize state
        state = self.sarsa.discretize_state(0, self.params.max_iterations, 0, 1.0)

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # Handle stagnation via perturbation
            if self.stagnation_count >= self.sarsa.stagnation_thresholds[1]:
                p_idx = self.random.randint(0, len(self.perturbation_ops) - 1)
                current_routes = self.perturbation_ops[p_idx](current_routes)
                current_cost = self.calculate_cost(current_routes)
                current_rev = sum(self.wastes.get(n, 0) * self.R for r in current_routes for n in r)
                current_profit = current_rev - current_cost
                current_eval_profit = self.evaluator(current_routes) if self.evaluator else current_profit
                self.stagnation_count = 0

                # Update best if perturbed route somehow magically improves it
                if current_profit > best_profit:
                    best_profit = current_profit
                    best_routes = copy.deepcopy(current_routes)
                    best_cost = current_cost

            # Calculate diversity (simple measure: variation in route sizes)
            diversity = self._calculate_diversity(current_routes)
            self.diversity_history.append(diversity)

            # SARSA: Select action (operator pair) from current state
            d_idx, r_idx = self.sarsa.select_operators(state, self.random)

            # Apply destroy-repair
            new_routes, removed = self.destroy_ops[d_idx](
                copy.deepcopy(current_routes), self._calc_removal_size(current_routes)
            )
            new_routes = self.repair_ops[r_idx](new_routes, removed)

            # Evaluate new solution
            new_cost = self.calculate_cost(new_routes)
            new_rev = sum(self.wastes.get(n, 0) * self.R for r in new_routes for n in r)
            new_profit = new_rev - new_cost

            new_eval_profit = self.evaluator(new_routes) if self.evaluator else new_profit

            # Acceptance criterion (simulated annealing)
            delta = current_eval_profit - new_eval_profit
            accept = False
            if delta < self.improvement_thresholds[0]:
                accept = True
                score = 3.0  # New best
            elif T > 0 and self.random.random() < math.exp(-delta / T):
                accept = True
                score = 1.0  # Accepted but not best
            else:
                score = 0.0  # Rejected

            # Calculate reward for SARSA
            if new_profit > best_profit + self.improvement_thresholds[1]:
                reward = 10.0  # Global best
                self.stagnation_count = 0
            elif new_profit > current_profit:
                reward = 5.0  # Local improvement
                self.stagnation_count = 0
            elif accept:
                reward = 1.0  # Diversification
                self.stagnation_count += 1
            else:
                reward = -1.0  # Rejection
                self.stagnation_count += 1

            # SARSA: Observe next state and select next action
            next_state = self.sarsa.discretize_state(
                iteration + 1, self.params.max_iterations, self.stagnation_count, diversity
            )
            next_d_idx, next_r_idx = self.sarsa.select_operators(next_state, self.random)
            next_action = (next_d_idx, next_r_idx)

            # SARSA: Update Q-value
            self.sarsa.update_q_value(state, (d_idx, r_idx), reward, next_state, next_action)

            # Update current solution if accepted
            if accept:
                current_routes = new_routes
                current_profit = new_profit
                current_eval_profit = new_eval_profit

                if new_profit > best_profit + self.improvement_thresholds[1]:
                    best_routes = copy.deepcopy(new_routes)
                    best_profit = new_profit
                    best_cost = new_cost

            # SARSA: Transition to next state
            state = next_state

            # Cool temperature
            T *= self.params.cooling_rate

            # Decay exploration
            if iteration % self.epsilon_decay_step == 0:
                self.sarsa.decay_epsilon()

            # Visualization
            self._viz_record(
                iteration=iteration,
                destroy_op=self.destroy_names[d_idx],
                repair_op=self.repair_names[r_idx],
                best_profit=best_profit,
                current_profit=current_profit,
                temperature=T,
                accepted=int(accept),
                score=score,
                reward=reward,
                epsilon=self.sarsa.epsilon,
                stagnation=self.stagnation_count,
                diversity=diversity,
            )

        return best_routes, best_profit, best_cost

    # ===== Helper Methods =====

    def _initialize_solve(
        self, initial_solution: Optional[List[List[int]]]
    ) -> Tuple[List[List[int]], List[List[int]], float, float]:
        """Initialize solution and metrics."""
        current_routes = initial_solution or self.build_initial_solution()
        best_routes = copy.deepcopy(current_routes)

        best_cost = self.calculate_cost(best_routes)
        best_rev = sum(self.wastes.get(n, 0) * self.R for r in best_routes for n in r)
        best_profit = best_rev - best_cost

        return current_routes, best_routes, best_profit, best_cost

    def _calc_removal_size(self, routes: List[List[int]]) -> int:
        """Calculate number of nodes to remove based on current solution size."""
        current_n_nodes = sum(len(r) for r in routes)

        if current_n_nodes == 0:
            return 0

        # Scale removal percentage with minimum of 2 nodes
        lower_bound = min(current_n_nodes, 2)
        max_pct_remove = int(current_n_nodes * self.params.max_removal_pct)
        upper_bound = max(lower_bound + 1, max_pct_remove)
        upper_bound = min(upper_bound, current_n_nodes)

        return self.random.randint(lower_bound, upper_bound)

    def _calculate_diversity(self, routes: List[List[int]]) -> float:
        """Calculate solution diversity (simple measure based on route variation)."""
        if not routes:
            return 0.0

        sizes = [len(r) for r in routes if r]
        if not sizes:
            return 0.0

        # Coefficient of variation
        mean_size = np.mean(sizes)
        if mean_size == 0:
            return 0.0

        std_size = np.std(sizes)
        cv = std_size / mean_size

        # Normalize to [0, 1]
        return min(1.0, cv)

    def calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculate total routing cost."""
        total_dist = 0.0
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
        """Build initial feasible solution using greedy heuristic."""
        nodes = self.nodes[:]
        self.random.shuffle(nodes)
        routes = []
        curr_route = []
        load = 0.0
        mandatory_set = set(self.mandatory_nodes)
        for node in nodes:
            waste = self.wastes.get(node, 0)
            revenue = waste * self.R
            is_mandatory = node in mandatory_set

            # VRPP check: is node profitable?
            if not is_mandatory and revenue < (self.dist_matrix[0][node] + self.dist_matrix[node][0]) * self.C:
                continue

            if load + waste <= self.capacity:
                curr_route.append(node)
                load += waste
            else:
                if curr_route:
                    routes.append(curr_route)
                curr_route = [node]
                load = waste

        if curr_route:
            routes.append(curr_route)

        return routes
