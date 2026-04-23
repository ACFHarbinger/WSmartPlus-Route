"""ALNS Policy wrapper for RL4CO integration.

This module provides the `VectorizedALNS` class, which wraps the high-performance
ALNS engine to satisfy the `AutoregressivePolicy` interface used in training
and evaluation pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict

from logic.src.constants.simulation import VEHICLE_CAPACITY
from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.autoregressive.policy import AutoregressivePolicy
from logic.src.models.policies.hgs import VectorizedHGS

from .adaptive_large_neighborhood_search import (
    VectorizedALNS as VectorizedALNSEngine,
)


class VectorizedALNS(AutoregressivePolicy):
    """RL4CO-compatible ALNS Policy wrapper.

    Orchestrates the data extraction from environment states (TensorDict),
    execution of the vectorized ALNS engine, and conversion of the resulting
    routes back into a standardized action format.

    Attributes:
        time_limit: Execution timeout for the solver in seconds.
        max_iterations: Internal search loop limit.
        max_vehicles: Fleet size limit for discretization.
        start_temp: Initial temperature for Simulated Annealing.
        cooling_rate: SA temperature decay factor.
    """

    def __init__(
        self,
        env_name: str,
        time_limit: float = 5.0,
        max_iterations: int = 500,
        max_vehicles: int = 0,
        start_temp: float = 0.5,
        cooling_rate: float = 0.9995,
        device: str = "cpu",
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialize the ALNS policy wrapper.

        Args:
            env_name: Identifier for the target RL4CO environment.
            time_limit: Search timeout in seconds.
            max_iterations: Number of improvement steps per instance.
            max_vehicles: Fleet size constraint for the split operator.
            start_temp: Initial SA temperature.
            cooling_rate: Decay rate for SA cooling.
            device: Hardware target.
            seed: Random initialization constant.
            **kwargs: Additional policy parameters.
        """
        super().__init__(env_name=env_name, device=device, seed=seed, **kwargs)
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.max_vehicles = max_vehicles
        self.start_temp = start_temp
        self.cooling_rate = cooling_rate

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        strategy: str = "greedy",
        num_starts: int = 1,
        max_steps: Optional[int] = None,
        phase: str = "train",
        return_actions: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute the ALNS solver on a batch of environment instances.

        Extracts locations, wastes, and capacities from the TensorDict, executes
        the vectorized ALNS optimization, and computes rewards for the resulting
        action sequences.

        Args:
            td: State bundle containing problem nodes and constraints.
            env: Relevant environment instance for reward calculation.
            strategy: Search strategy (ignored; ALNS is meta-heuristic).
            num_starts: Number of restarts (not applicable).
            max_steps: Sequence length limit (ignored).
            phase: Current execution phase ('train', 'val', or 'test').
            return_actions: Whether to include action tensors in the output.
            **kwargs: Execution-time overrides (e.g., max_vehicles).

        Returns:
            Dict[str, Any]: Results including:
                - reward (torch.Tensor): Calculated reward/cost for the solution.
                - actions (torch.Tensor): The sequence of applied moves.
                - log_likelihood (torch.Tensor): Zero vector (ALNS is non-pi).
                - cost (torch.Tensor): Raw objective values from the engine.
        """
        batch_size = td.batch_size[0]
        device = td.device if td.device is not None else td["locs"].device

        # 1. Data Parsing
        customers = td["locs"]  # (batch, num_nodes, 2)
        depot = td["depot"].unsqueeze(1)  # (batch, 1, 2)
        locs = torch.cat([depot, customers], dim=1)  # (batch, num_nodes + 1, 2)

        num_nodes = locs.shape[1]

        # Compute Euclidean distance matrix for the engine
        if locs.dim() == 3 and locs.shape[-1] == 2:
            diff = locs.unsqueeze(2) - locs.unsqueeze(1)
            dist_matrix = torch.sqrt((diff**2).sum(dim=-1))
        else:
            dist_matrix = locs

        waste_at_nodes = td.get("waste", torch.zeros(batch_size, num_nodes - 1, device=device))
        # Prepend 0/None waste for the depot node
        waste = torch.cat([torch.zeros(batch_size, 1, device=device), waste_at_nodes], dim=1)

        capacity = td.get("capacity", torch.ones(batch_size, device=device) * VEHICLE_CAPACITY)
        if capacity.dim() == 0:
            capacity = capacity.expand(batch_size)

        # 2. Engine Execution
        initial_solutions = torch.stack(
            [torch.randperm(num_nodes - 1, device=device, generator=self.generator) + 1 for _ in range(batch_size)]
        )

        solver = VectorizedALNSEngine(
            dist_matrix=dist_matrix,
            wastes=waste,
            vehicle_capacity=capacity,
            time_limit=self.time_limit,
            device=str(device),
            generator=self.generator,
        )

        routes_list, costs = solver.solve(
            initial_solutions=initial_solutions,
            n_iterations=self.max_iterations,
            time_limit=self.time_limit,
            max_vehicles=kwargs.get("max_vehicles", self.max_vehicles),
            start_temp=self.start_temp,
            cooling_rate=self.cooling_rate,
        )

        # 3. Output Packaging
        all_actions = []
        for b in range(batch_size):
            route_nodes_raw = routes_list[b]
            if isinstance(route_nodes_raw, torch.Tensor):
                route_tensor = route_nodes_raw
            else:
                route_tensor = torch.tensor(route_nodes_raw, device=device)

            actions = route_tensor if route_tensor.size(0) > 0 else torch.tensor([0, 0], device=device)
            all_actions.append(actions)

        # Pad variable-length action sequences for batch reward calculation
        max_len = max(len(a) for a in all_actions)
        padded_actions = torch.stack(
            [torch.cat([a, torch.zeros(max_len - len(a), device=device, dtype=torch.long)]) for a in all_actions]
        )

        reward = VectorizedHGS._compute_reward(td, env, padded_actions)

        return {
            "reward": reward,
            "actions": padded_actions,
            "log_likelihood": torch.zeros(batch_size, device=device),
            "cost": costs,
        }
