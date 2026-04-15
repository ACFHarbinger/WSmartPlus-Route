"""
Gymnasium Environment for DR-ALNS Training.

Wraps ALNS as a Gymnasium environment for PPO training.

Reference:
    Reijnen, R., Zhang, Y., Lau, H. C., & Bukhsh, Z.
    "Online Control of Adaptive Large Neighborhood Search Using Deep
    Reinforcement Learning", AAAI 2024.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from logic.src.models.core.dr_alns.ppo_agent import DRALNSState
from logic.src.policies.helpers.operators import (
    cluster_removal,
    greedy_insertion,
    random_removal,
    regret_2_insertion,
    worst_removal,
)
from logic.src.policies.helpers.operators.heuristics.nearest_neighbor_initialization import (
    build_nn_routes,
)


class DRALNSEnv(gym.Env):
    """
    Gymnasium environment for training DR-ALNS with PPO.

    The environment wraps ALNS and exposes it as a reinforcement learning
    problem where the agent learns to select operators and configure parameters.

    Action Space:
        MultiDiscrete([n_destroy, n_repair, 10, 50])
        - A1: Destroy operator index (0-2)
        - A2: Repair operator index (0-1)
        - A3: Destroy severity level (0-9 = 10%-100%)
        - A4: Temperature level (0-49 = 0.1-5.0)

    Observation Space:
        Box(7,) - 7 problem-agnostic features
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        max_iterations: int = 100,
        n_destroy_ops: int = 3,
        n_repair_ops: int = 2,
        instance_generator: Optional[Any] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize DR-ALNS environment.

        Args:
            max_iterations: Maximum iterations per episode.
            n_destroy_ops: Number of destroy operators (default 3).
            n_repair_ops: Number of repair operators (default 2).
            instance_generator: Callable that generates problem instances.
            seed: Random seed.
        """
        super().__init__()

        self.max_iterations = max_iterations
        self.n_destroy_ops = n_destroy_ops
        self.n_repair_ops = n_repair_ops
        self.instance_generator = instance_generator

        # Gymnasium spaces
        self.action_space = gym.spaces.MultiDiscrete([n_destroy_ops, n_repair_ops, 10, 50])
        self.observation_space = gym.spaces.Box(low=-1.0, high=100.0, shape=(7,), dtype=np.float32)

        # Problem instance data
        self.dist_matrix: Optional[np.ndarray] = None
        self.wastes: Optional[Dict[int, float]] = None
        self.capacity: Optional[float] = None
        self.R: float = 1.0  # Revenue multiplier
        self.C: float = 1.0  # Cost multiplier
        self.mandatory_nodes: List[int] = []
        self.nodes: List[int] = []

        # ALNS state
        self.current_routes: List[List[int]] = []
        self.best_routes: List[List[int]] = []
        self.current_profit: float = 0.0
        self.best_profit: float = 0.0
        self.previous_profit: float = 0.0

        # Episode tracking
        self.iteration: int = 0
        self.iterations_since_best: int = 0
        self.done: bool = False

        # State representation
        self.state = DRALNSState()

        # Random state
        self.rng = random.Random(seed)

        # Operators (matching reference implementation)
        self.destroy_ops = [
            self._random_removal,
            self._worst_removal,
            self._cluster_removal,
        ]
        self.repair_ops = [
            self._greedy_insertion,
            self._regret_2_insertion,
        ]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed.
            options: Additional options (can contain problem instance data).

        Returns:
            Tuple of (observation, info).
        """
        if seed is not None:
            self.rng = random.Random(seed)

        # Generate or load problem instance
        if options is not None and "instance" in options:
            # Load provided instance
            instance_data = options["instance"]
            self.dist_matrix = instance_data["dist_matrix"]
            self.wastes = instance_data["wastes"]
            self.capacity = instance_data["capacity"]
            self.R = instance_data.get("R", 1.0)
            self.C = instance_data.get("C", 1.0)
            self.mandatory_nodes = instance_data.get("mandatory_nodes", [])
        elif self.instance_generator is not None:
            # Generate new instance
            instance_data = self.instance_generator()
            self.dist_matrix = instance_data["dist_matrix"]
            self.wastes = instance_data["wastes"]
            self.capacity = instance_data["capacity"]
            self.R = instance_data.get("R", 1.0)
            self.C = instance_data.get("C", 1.0)
            self.mandatory_nodes = instance_data.get("mandatory_nodes", [])
        else:
            raise ValueError("No instance provided and no generator configured")

        # Setup problem
        n_nodes = len(self.dist_matrix) - 1
        self.nodes = list(range(1, n_nodes + 1))

        # Build initial solution
        self.current_routes = self._build_initial_solution()
        self.best_routes = copy.deepcopy(self.current_routes)

        self.current_profit = self._evaluate(self.current_routes)
        self.best_profit = self.current_profit
        self.previous_profit = self.current_profit

        # Reset episode tracking
        self.iteration = 0
        self.iterations_since_best = 0
        self.done = False

        # Initialize state
        self.state.update(
            best_profit=self.best_profit,
            current_profit=self.current_profit,
            previous_profit=self.previous_profit,
            new_accepted=True,
            new_best=False,
            iteration=0,
            max_iterations=self.max_iterations,
            iterations_since_best=0,
        )

        obs = self._get_observation()
        info = {"best_profit": self.best_profit}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action array [destroy_idx, repair_idx, severity_idx, temp_idx].

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        self.iteration += 1
        self.iterations_since_best += 1

        # Extract actions
        destroy_idx = int(action[0])
        repair_idx = int(action[1])
        severity_idx = int(action[2])
        temp_idx = int(action[3])

        # Convert indices to parameters
        severity = (severity_idx + 1) / 10.0  # 0.1 to 1.0
        temperature = (temp_idx + 1) * 0.1  # 0.1 to 5.0

        # Store previous profit for state update
        self.previous_profit = self.current_profit

        # Apply destroy and repair
        try:
            new_routes = self._apply_operators(self.current_routes, destroy_idx, repair_idx, severity)
            new_profit = self._evaluate(new_routes)
        except Exception:
            # If operators fail, don't change solution
            new_routes = self.current_routes
            new_profit = self.current_profit

        # Acceptance criterion (Simulated Annealing)
        accepted = self._accept(self.current_profit, new_profit, temperature)

        # Update solutions
        if accepted:
            self.current_routes = new_routes
            self.current_profit = new_profit

        # Check for new best and calculate reward
        reward = 0.0
        if self.current_profit > self.best_profit:
            self.best_routes = copy.deepcopy(self.current_routes)
            self.best_profit = self.current_profit
            reward = 5.0  # New global best
            self.iterations_since_best = 0
            new_best = True
        elif accepted:
            reward = 1.0 if self.current_profit > self.previous_profit else 0.1
            self.iterations_since_best += 1
            new_best = False
        else:
            reward = -0.1  # Rejected
            self.iterations_since_best += 1
            new_best = False

        # Update state
        self.state.update(
            best_profit=self.best_profit,
            current_profit=self.current_profit,
            previous_profit=self.previous_profit,
            new_accepted=accepted,
            new_best=new_best,
            iteration=self.iteration,
            max_iterations=self.max_iterations,
            iterations_since_best=self.iterations_since_best,
        )

        # Check termination
        terminated = self.iteration >= self.max_iterations
        truncated = False

        obs = self._get_observation()
        info = {
            "best_profit": self.best_profit,
            "current_profit": self.current_profit,
            "iteration": self.iteration,
            "accepted": accepted,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        return np.array(
            [
                float(self.state.best_improved),
                float(self.state.current_accepted),
                float(self.state.current_improved),
                float(self.state.is_current_best),
                self.state.cost_diff_best,
                float(self.state.stagnation_count),
                self.state.search_budget,
            ],
            dtype=np.float32,
        )

    def _apply_operators(
        self,
        routes: List[List[int]],
        destroy_idx: int,
        repair_idx: int,
        severity: float,
    ) -> List[List[int]]:
        """Apply selected destroy and repair operators."""
        total_nodes = sum(len(route) for route in routes)
        if total_nodes == 0:
            return routes

        n_remove = max(1, int(total_nodes * severity))

        # Apply destroy
        destroy_op = self.destroy_ops[destroy_idx]
        partial_routes, removed_nodes = destroy_op(copy.deepcopy(routes), n_remove)

        # Apply repair
        repair_op = self.repair_ops[repair_idx]
        repaired_routes = repair_op(partial_routes, removed_nodes)

        return repaired_routes

    def _accept(self, current_profit: float, new_profit: float, temperature: float) -> bool:
        """Simulated Annealing acceptance criterion."""
        if new_profit >= current_profit:
            return True
        if temperature <= 0:
            return False
        delta = new_profit - current_profit
        probability = np.exp(delta / temperature)
        return self.rng.random() < probability

    # Destroy operators
    def _random_removal(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        return random_removal(routes, n, rng=self.rng)

    def _worst_removal(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        assert self.dist_matrix is not None
        return worst_removal(routes, n, self.dist_matrix)

    def _cluster_removal(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        assert self.dist_matrix is not None
        return cluster_removal(routes, n, self.dist_matrix, self.nodes, rng=self.rng)

    # Repair operators
    def _greedy_insertion(self, partial_routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        assert self.dist_matrix is not None and self.wastes is not None and self.capacity is not None
        return greedy_insertion(
            partial_routes,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _regret_2_insertion(self, partial_routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        assert self.dist_matrix is not None and self.wastes is not None and self.capacity is not None
        return regret_2_insertion(
            partial_routes,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
        )

    # Helpers
    def _build_initial_solution(self) -> List[List[int]]:
        assert self.dist_matrix is not None and self.wastes is not None and self.capacity is not None
        return build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
        )

    def _evaluate(self, routes: List[List[int]]) -> float:
        assert self.wastes is not None
        if not routes:
            return 0.0
        revenue = sum(self.wastes.get(node, 0.0) * self.R for route in routes for node in route)
        cost = self._cost(routes) * self.C
        return revenue - cost

    def _cost(self, routes: List[List[int]]) -> float:
        assert self.dist_matrix is not None
        total_cost = 0.0
        for route in routes:
            if not route:
                continue
            total_cost += self.dist_matrix[0, route[0]]
            for i in range(len(route) - 1):
                total_cost += self.dist_matrix[route[i], route[i + 1]]
            total_cost += self.dist_matrix[route[-1], 0]
        return total_cost
