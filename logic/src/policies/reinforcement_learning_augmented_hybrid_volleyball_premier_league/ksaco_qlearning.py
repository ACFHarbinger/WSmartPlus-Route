"""
Enhanced K-Sparse ACO Solver with Q-Learning for Local Search Operator Selection.

This module extends the standard K-Sparse ACO with Q-Learning reinforcement learning
to dynamically select the most effective local search operators for route improvement.

Local Search Operators:
    Exchange:
        - λ-interchange (cross-exchange)
        - OR-opt (relocate chains)
        - Ejection chains

    Route:
        - 2-opt* (inter-route tail exchange)
        - SWAP* (inter-route node swap)
        - 2-opt (intra-route reversal)
        - 3-opt (intra-route reconnection)

    Move:
        - Relocate single nodes
        - Swap within route

    Heuristics:
        - Lin-Kernighan-Helsgaun (LKH) for TSP sub-problems

Reference:
    Watkins & Dayan, "Q-Learning", Machine Learning, 1992.
    Gambardella & Dorigo, "Ant Colony System", IEEE Trans., 1997.
"""

import random
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..ant_colony_optimization.k_sparse_aco.construction import SolutionConstructor
from ..ant_colony_optimization.k_sparse_aco.params import ACOParams
from ..ant_colony_optimization.k_sparse_aco.pheromones import SparsePheromoneTau

# Import operators from their respective modules
from ..operators.exchange import cross_exchange, ejection_chain, lambda_interchange, move_or_opt
from ..operators.heuristics import build_nn_routes, solve_lkh
from ..operators.move import move_relocate, move_swap
from ..operators.route import move_2opt_intra, move_2opt_star, move_3opt_intra, move_swap_star
from .params import RLAHVPLParams


class QLearningOperatorSelector:
    """
    Q-Learning for local search operator selection.

    Unlike SARSA (on-policy), Q-Learning is off-policy and learns
    the optimal action-value function directly.
    """

    def __init__(
        self,
        n_operators: int,
        alpha: float = 0.15,  # Learning rate
        gamma: float = 0.9,  # Discount factor
        epsilon: float = 0.25,  # Initial exploration rate
        epsilon_decay: float = 0.99,
        epsilon_min: float = 0.05,
        rewards_size: int = 20,
    ):
        """Initialize Q-Learning operator selector."""
        self.n_operators = n_operators
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: Q[state][action] = expected cumulative reward
        # State: (improvement_rate, route_quality, iteration_phase)
        self.q_table: Dict[Tuple[int, int, int], np.ndarray] = defaultdict(lambda: np.zeros(n_operators))

        # Performance tracking
        self.op_uses: Dict[int, int] = defaultdict(int)
        self.op_rewards: Dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=rewards_size))
        self.op_improvements: Dict[int, int] = defaultdict(int)

    def discretize_state(
        self, improvement_rate: float, route_quality: float, iteration_progress: float
    ) -> Tuple[int, int, int]:
        """
        Discretize continuous state into bins.

        Args:
            improvement_rate: Recent rate of cost improvement (0-1).
            route_quality: Solution quality measure (0-1).
            iteration_progress: Progress through iterations (0-1).

        Returns:
            Discretized state tuple.
        """
        # Improvement rate: low (0), medium (1), high (2)
        imp_level = 0 if improvement_rate < 0.2 else (1 if improvement_rate < 0.5 else 2)

        # Route quality: poor (0), average (1), good (2)
        qual_level = 0 if route_quality < 0.4 else (1 if route_quality < 0.7 else 2)

        # Phase: early (0), mid (1), late (2)
        phase = 0 if iteration_progress < 0.33 else (1 if iteration_progress < 0.67 else 2)

        return (imp_level, qual_level, phase)

    def select_operator(self, state: Tuple[int, int, int], rng: Any) -> int:
        """
        Select operator using ε-greedy policy.

        Args:
            state: Current state.
            rng: Random number generator.

        Returns:
            Operator index.
        """
        if rng.random() < self.epsilon:
            # Explore: random operator
            return rng.randint(0, self.n_operators - 1)
        else:
            # Exploit: best operator from Q-table
            q_values = self.q_table[state]
            max_q = np.max(q_values)

            # Handle ties
            best_ops = np.where(q_values == max_q)[0]
            return int(rng.choice(best_ops)) if len(best_ops) > 0 else 0

    def update_q_value(
        self,
        state: Tuple[int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int],
    ):
        """
        Update Q-value using Q-learning temporal difference update.

        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])

        # Q-learning update (off-policy)
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.alpha * td_error

        self.q_table[state][action] = new_q

        # Track performance
        self.op_uses[action] += 1
        self.op_rewards[action].append(reward)
        if reward > 0:
            self.op_improvements[action] += 1

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_best_operators(self, top_k: int = 5) -> List[Tuple[int, float]]:
        """Get operators with highest average rewards."""
        avg_rewards = {}
        for op_id, rewards in self.op_rewards.items():
            if rewards:
                avg_rewards[op_id] = np.mean(rewards)

        sorted_ops = sorted(avg_rewards.items(), key=lambda x: x[1], reverse=True)
        return sorted_ops[:top_k]


class LocalSearchManager:
    """
    Manages local search operators for ACO route improvement.

    This class provides a unified interface to all local search operators
    from the operators module, compatible with Q-Learning operator selection.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        improvement_threshold: float,
        seed: Optional[int] = None,
    ):
        """Initialize local search manager."""
        self.d = dist_matrix
        self.waste = wastes
        self.Q = capacity
        self.R = R
        self.C = C
        self.improvement_threshold = improvement_threshold
        self.rng = np.random.default_rng(seed)
        self.random_std = random.Random(seed)

        # Routes interface for operators
        self.routes: List[List[int]] = []

        # Load cache for efficiency
        self._load_cache: Dict[int, float] = {}

    def set_routes(self, routes: List[List[int]]):
        """Set current routes for local search."""
        self.routes = [r[:] for r in routes]
        self._invalidate_cache()

    def get_routes(self) -> List[List[int]]:
        """Get current routes."""
        return [r[:] for r in self.routes]

    def _calc_load_fresh(self, route: List[int]) -> float:
        """Calculate route load."""
        return sum(self.waste.get(n, 0) for n in route)

    def _get_load_cached(self, r_idx: int) -> float:
        """Get cached load for route."""
        if r_idx not in self._load_cache:
            self._load_cache[r_idx] = self._calc_load_fresh(self.routes[r_idx])
        return self._load_cache[r_idx]

    def _update_map(self, route_indices: set):
        """Update load cache after route modifications."""
        for r_idx in route_indices:
            if r_idx < len(self.routes):
                self._load_cache[r_idx] = self._calc_load_fresh(self.routes[r_idx])

    def _invalidate_cache(self):
        """Invalidate all cached loads."""
        self._load_cache.clear()

    # ===== Exchange Operators =====

    def or_opt(self, chain_len: int = 2) -> bool:
        """
        OR-opt: relocate chain of nodes using the imported operator.

        Tries all possible OR-opt moves and applies the first improving one.
        """
        if not self.routes:
            return False

        for r_idx, route in enumerate(self.routes):
            if len(route) < chain_len:
                continue

            for pos in range(len(route)):
                if pos + chain_len > len(route):
                    continue

                node = route[pos]
                if move_or_opt(self, node, chain_len, r_idx, pos):
                    return True

        return False

    def cross_exchange_op(self, max_seg_len: int = 2) -> bool:
        """
        Cross-exchange: swap segments between routes.

        Systematically tries cross-exchange moves between route pairs.
        """
        if len(self.routes) < 2:
            return False

        for r_a in range(len(self.routes)):
            for r_b in range(r_a + 1, len(self.routes)):
                for seg_a_len in range(max_seg_len + 1):
                    for seg_b_len in range(max_seg_len + 1):
                        if seg_a_len == 0 and seg_b_len == 0:
                            continue

                        for seg_a_start in range(len(self.routes[r_a]) - seg_a_len + 1):
                            for seg_b_start in range(len(self.routes[r_b]) - seg_b_len + 1):
                                if cross_exchange(self, r_a, seg_a_start, seg_a_len, r_b, seg_b_start, seg_b_len):
                                    return True
        return False

    def lambda_interchange_op(self, lambda_max: int = 2) -> bool:
        """
        λ-interchange: generalized cross-exchange with segment lengths up to λ.
        """
        return lambda_interchange(self, lambda_max)

    def ejection_chain_op(self, max_depth: int = 3) -> bool:
        """
        Ejection chain: attempt to empty routes for fleet minimization.
        """
        if len(self.routes) < 2:
            return False

        # Try to eject from smallest routes first
        route_sizes = [(i, len(r)) for i, r in enumerate(self.routes) if r]
        route_sizes.sort(key=lambda x: x[1])

        # Try ejection on up to 3 smallest routes
        return any(ejection_chain(self, r_idx, max_depth) for r_idx, _ in route_sizes[:3])

    # ===== Route Operators =====

    def two_opt_star(self) -> bool:
        """2-opt* inter-route operator using imported implementation."""
        if len(self.routes) < 2:
            return False

        for r_u in range(len(self.routes)):
            for r_v in range(r_u + 1, len(self.routes)):
                route_u = self.routes[r_u]
                route_v = self.routes[r_v]

                if not route_u or not route_v:
                    continue

                for p_u in range(len(route_u)):
                    u = route_u[p_u]
                    for p_v in range(len(route_v)):
                        v = route_v[p_v]
                        if move_2opt_star(self, u, v, r_u, p_u, r_v, p_v):
                            return True

        return False

    def swap_star(self) -> bool:
        """SWAP* inter-route operator using imported implementation."""
        if len(self.routes) < 2:
            return False

        for r_u in range(len(self.routes)):
            for r_v in range(r_u + 1, len(self.routes)):
                route_u = self.routes[r_u]
                route_v = self.routes[r_v]

                if not route_u or not route_v:
                    continue

                for p_u in range(len(route_u)):
                    u = route_u[p_u]
                    for p_v in range(len(route_v)):
                        v = route_v[p_v]
                        if move_swap_star(self, u, v, r_u, p_u, r_v, p_v):
                            return True

        return False

    def two_opt_intra(self) -> bool:
        """2-opt intra-route: reverse segments within routes."""
        for r_u in range(len(self.routes)):
            route = self.routes[r_u]
            if len(route) < 3:
                continue

            for p_u in range(len(route) - 1):
                u = route[p_u]
                for p_v in range(p_u + 2, len(route)):
                    v = route[p_v]
                    if move_2opt_intra(self, u, v, r_u, p_u, r_u, p_v):
                        return True
        return False

    def three_opt_intra(self) -> bool:
        """3-opt intra-route: reconnect three segments within routes."""
        for r_u in range(len(self.routes)):
            route = self.routes[r_u]
            if len(route) < 4:
                continue

            for p_u in range(len(route) - 2):
                u = route[p_u]
                for p_v in range(p_u + 2, len(route)):
                    v = route[p_v]
                    if move_3opt_intra(self, u, v, r_u, p_u, r_u, p_v, self.random_std):
                        return True
        return False

    # ===== Move Operators =====

    def relocate(self) -> bool:
        """Relocate: move single nodes to better positions."""
        for r_u in range(len(self.routes)):
            route_u = self.routes[r_u]
            if not route_u:
                continue

            for p_u in range(len(route_u)):
                u = route_u[p_u]

                # Try relocating to all routes
                for r_v in range(len(self.routes)):
                    route_v = self.routes[r_v]

                    for p_v in range(len(route_v)):
                        v = route_v[p_v]
                        if move_relocate(self, u, v, r_u, p_u, r_v, p_v):
                            return True
        return False

    def swap(self) -> bool:
        """Swap: exchange positions of two nodes."""
        for r_u in range(len(self.routes)):
            route_u = self.routes[r_u]
            if not route_u:
                continue

            for p_u in range(len(route_u)):
                u = route_u[p_u]

                # Try swapping with nodes in all routes
                for r_v in range(r_u, len(self.routes)):
                    route_v = self.routes[r_v]
                    start_p_v = p_u + 1 if r_v == r_u else 0

                    for p_v in range(start_p_v, len(route_v)):
                        v = route_v[p_v]
                        if move_swap(self, u, v, r_u, p_u, r_v, p_v):
                            return True
        return False

    # ===== Heuristic Operators =====

    def lkh_refinement(self) -> bool:
        """
        Apply Lin-Kernighan-Helsgaun heuristic to refine individual routes.

        Uses LKH to optimize each route as a TSP subproblem.
        """
        improved = False
        waste_array = np.zeros(len(self.d))
        for node, w in self.waste.items():
            if node < len(waste_array):
                waste_array[node] = w

        for r_idx, route in enumerate(self.routes):
            if len(route) < 3:
                continue

            # Build TSP instance for this route
            nodes = [0] + route + [0]
            initial_cost = sum(self.d[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1))

            # Apply LKH
            improved_tour, improved_cost = solve_lkh(
                self.d, initial_tour=nodes, max_iterations=20, waste=waste_array, capacity=self.Q, np_rng=self.rng
            )

            if improved_cost < initial_cost - 1e-6:
                # Extract route from tour (remove depot duplicates)
                new_route = [n for n in improved_tour if n != 0]
                self.routes[r_idx] = new_route
                self._update_map({r_idx})
                improved = True

        return improved


class KSparseACOQLSolver(PolicyVizMixin):
    """
    Enhanced K-Sparse ACO with Q-Learning for local search operator selection.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: ACOParams,
        rl_params: RLAHVPLParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize enhanced ACO-QL solver."""
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes
        self.epsilon_decay_step = rl_params.qlearning_epsilon_decay_step
        self.improvement_threshold = rl_params.qlearning_improvement_thresholds[0]

        self.n_nodes = len(dist_matrix)
        self.nodes = list(range(1, self.n_nodes))

        # Pheromone and heuristic
        self.eta = np.zeros_like(dist_matrix, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.eta = np.where(dist_matrix > 0, 1.0 / dist_matrix, 0.0)

        nn_cost = self._nearest_neighbor_cost()
        self.tau_0 = (
            params.tau_0
            if params.tau_0 is not None and params.tau_0 > 0
            else 1.0 / (self.n_nodes * nn_cost)
            if nn_cost > 0
            else params.tau_max
        )

        self.pheromone = SparsePheromoneTau(self.n_nodes, params.k_sparse, self.tau_0, params.tau_min, params.tau_max)

        # Local search
        self.ls_manager = LocalSearchManager(
            dist_matrix, wastes, capacity, R, C, rl_params.qlearning_improvement_thresholds[1], seed=seed
        )

        # Candidate lists
        self.candidate_lists = self._build_candidate_lists()

        # Constructor
        self.constructor = SolutionConstructor(
            dist_matrix,
            wastes,
            capacity,
            self.pheromone,
            self.eta,
            self.candidate_lists,
            self.nodes,
            params,
            self.tau_0,
            R=R,
            C=C,
            mandatory_nodes=mandatory_nodes,
            seed=seed,
        )

        # Q-Learning selector for local search operators
        # Full operator set across all categories
        self.ls_operators = [
            # Exchange operators
            "or_opt",
            "cross_exchange",
            "lambda_interchange",
            "ejection_chain",
            # Route operators
            "2opt_star",
            "swap_star",
            "2opt_intra",
            "3opt_intra",
            # Move operators
            "relocate",
            "swap",
            # Heuristic operators
            "lkh_refinement",
        ]
        self.q_learner = QLearningOperatorSelector(
            n_operators=len(self.ls_operators),
            alpha=rl_params.qlearning_alpha,
            gamma=rl_params.qlearning_gamma,
            epsilon=rl_params.qlearning_epsilon,
            epsilon_decay=rl_params.qlearning_epsilon_decay,
            epsilon_min=rl_params.qlearning_epsilon_min,
            rewards_size=rl_params.qlearning_rewards_size,
        )

        # Tracking
        self.improvement_history: Deque[float] = deque(maxlen=rl_params.qlearning_history_size)

    def _nearest_neighbor_cost(self) -> float:
        """Compute nearest neighbor heuristic cost."""
        visited = set([0])
        current = 0
        cost = 0.0
        for _ in range(len(self.nodes)):
            best_next = None
            best_dist = float("inf")
            for node in self.nodes:
                if node not in visited:
                    d = self.dist_matrix[current][node]
                    if d < best_dist:
                        best_dist = d
                        best_next = node
            if best_next is not None:
                cost += best_dist
                visited.add(best_next)
                current = best_next
        cost += self.dist_matrix[current][0]
        return cost

    def _build_candidate_lists(self) -> Dict[int, List[int]]:
        """Build k-nearest neighbor candidate lists."""
        candidates: Dict[int, List[int]] = {}
        k = min(self.params.k_sparse, len(self.nodes))

        for i in range(self.n_nodes):
            distances = [(self.dist_matrix[i][j], j) for j in range(self.n_nodes) if j != i]
            distances.sort()
            candidates[i] = [j for _, j in distances[:k]]

        return candidates

    def _initialize_with_nn_heuristic(self) -> Tuple[List[List[int]], float, float]:
        """
        Initialize solution using Nearest Neighbor heuristic.

        Creates geographically compact routes using build_nn_routes and applies
        local search to improve the initial solution.

        Returns:
            Tuple of (routes, profit, cost) for the NN-initialized solution.
        """
        # Build initial routes using NN heuristic
        nn_routes = build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes if self.mandatory_nodes else [],
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
            rng=self.constructor.random,
        )

        # Apply local search to improve NN solution
        if self.params.local_search and nn_routes:
            nn_routes = self._q_learning_local_search(nn_routes, iteration=0)

        # Calculate metrics
        cost = self._calculate_cost(nn_routes)
        revenue = sum(self.wastes.get(n, 0) * self.R for r in nn_routes for n in r)
        profit = revenue - cost

        return nn_routes, profit, cost

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run enhanced ACO with Q-Learning local search.

        The algorithm is seeded with a Nearest Neighbor heuristic solution
        to provide a strong starting point for the ACO search.
        """
        # Initialize with NN heuristic for strong starting solution
        best_routes, best_profit, best_cost = self._initialize_with_nn_heuristic()
        start_time = time.process_time()

        # Update pheromones based on NN solution
        if best_routes:
            self._global_pheromone_update(best_routes, best_cost)

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            iteration_best_routes: List[List[int]] = []
            iteration_best_profit = float("-inf")
            iteration_best_cost = float("inf")

            # Ant construction
            for _ in range(self.params.n_ants):
                routes = self.constructor.construct()

                # Q-Learning local search
                if self.params.local_search:
                    routes = self._q_learning_local_search(routes, iteration)

                cost = self._calculate_cost(routes)
                revenue = sum(self.wastes.get(n, 0) * self.R for r in routes for n in r)
                profit = revenue - cost

                if profit > iteration_best_profit:
                    iteration_best_profit = profit
                    iteration_best_cost = cost
                    iteration_best_routes = routes

            # Update global best
            if iteration_best_profit > best_profit:
                improvement = (iteration_best_profit - best_profit) / (abs(best_profit) + 1e-6)
                self.improvement_history.append(improvement)
                best_profit = iteration_best_profit
                best_cost = iteration_best_cost
                best_routes = iteration_best_routes
            else:
                self.improvement_history.append(0.0)

            # Global pheromone update
            self._global_pheromone_update(best_routes, best_cost)

            # Visualization
            self._viz_record(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                iter_best_profit=iteration_best_profit,
                epsilon=self.q_learner.epsilon,
            )

        return best_routes, best_profit, best_cost

    def _q_learning_local_search(self, routes: List[List[int]], iteration: int) -> List[List[int]]:
        """Apply Q-Learning guided local search with full operator set."""
        self.ls_manager.set_routes(routes)
        initial_cost = self._calculate_cost(routes)

        # State discretization
        improvement_rate = np.mean(self.improvement_history) if self.improvement_history else 0.5
        route_quality = 1.0 / (1.0 + initial_cost / 1000.0)  # Normalize
        iteration_progress = iteration / max(self.params.max_iterations, 1)

        state = self.q_learner.discretize_state(improvement_rate, route_quality, iteration_progress)

        # Apply multiple local search moves
        for _ in range(self.params.local_search_iterations):
            # Select operator via Q-Learning
            op_idx = self.q_learner.select_operator(state, self.constructor.random)
            operator_name = self.ls_operators[op_idx]

            # Apply selected operator
            improved = self._apply_operator(operator_name)

            # Calculate reward
            new_routes = self.ls_manager.get_routes()
            new_cost = self._calculate_cost(new_routes)
            cost_improvement = initial_cost - new_cost
            reward = 10.0 if cost_improvement > self.improvement_threshold else (-1.0 if not improved else 0.0)

            # Next state
            new_quality = 1.0 / (1.0 + new_cost / 1000.0)
            next_state = self.q_learner.discretize_state(
                cost_improvement / (initial_cost + 1e-6), new_quality, iteration_progress
            )

            # Q-Learning update
            self.q_learner.update_q_value(state, op_idx, reward, next_state)

            # Update state
            state = next_state
            if improved:
                initial_cost = new_cost

        # Decay exploration
        if iteration % self.epsilon_decay_step == 0:
            self.q_learner.decay_epsilon()

        return self.ls_manager.get_routes()

    def _apply_operator(self, operator_name: str) -> bool:
        """
        Apply the specified local search operator.

        Args:
            operator_name: Name of the operator to apply.

        Returns:
            True if the operator improved the solution.
        """
        # Exchange operators
        if operator_name == "or_opt":
            return self.ls_manager.or_opt(chain_len=2)
        elif operator_name == "cross_exchange":
            return self.ls_manager.cross_exchange_op(max_seg_len=2)
        elif operator_name == "lambda_interchange":
            return self.ls_manager.lambda_interchange_op(lambda_max=2)
        elif operator_name == "ejection_chain":
            return self.ls_manager.ejection_chain_op(max_depth=3)

        # Route operators
        elif operator_name == "2opt_star":
            return self.ls_manager.two_opt_star()
        elif operator_name == "swap_star":
            return self.ls_manager.swap_star()
        elif operator_name == "2opt_intra":
            return self.ls_manager.two_opt_intra()
        elif operator_name == "3opt_intra":
            return self.ls_manager.three_opt_intra()

        # Move operators
        elif operator_name == "relocate":
            return self.ls_manager.relocate()
        elif operator_name == "swap":
            return self.ls_manager.swap()

        # Heuristic operators
        elif operator_name == "lkh_refinement":
            return self.ls_manager.lkh_refinement()

        else:
            # Unknown operator
            return False

    def _global_pheromone_update(self, best_routes: List[List[int]], best_cost: float):
        """ACS global pheromone update."""
        if not best_routes or best_cost <= 0:
            return

        self.pheromone.evaporate_all(self.params.rho)
        delta = self.params.elitist_weight / best_cost

        for route in best_routes:
            if not route:
                continue
            self.pheromone.update_edge(0, route[0], delta, evaporate=False)
            for k in range(len(route) - 1):
                self.pheromone.update_edge(route[k], route[k + 1], delta, evaporate=False)
            self.pheromone.update_edge(route[-1], 0, delta, evaporate=False)

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculate total routing cost."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total * self.C
