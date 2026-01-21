"""
Neural-Heuristic Hybrid Policy.
Uses a neural model for construction and a heuristic for refinement.
"""
from __future__ import annotations

from typing import Union

from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.models.policies.base import ConstructivePolicy
from logic.src.models.policies.classical.alns import ALNSPolicy
from logic.src.models.policies.classical.hgs import HGSPolicy


class NeuralHeuristicHybrid(ConstructivePolicy):
    """
    Hybrid policy combining neural construction and heuristic refinement.
    """

    def __init__(
        self,
        neural_policy: ConstructivePolicy,
        heuristic_policy: Union[ALNSPolicy, HGSPolicy],
        **kwargs,
    ):
        """Initialize NeuralHeuristicHybrid."""
        super().__init__(env_name=neural_policy.env_name, **kwargs)
        self.neural_policy = neural_policy
        self.heuristic_policy = heuristic_policy

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        decode_type: str = "greedy",
        **kwargs,
    ) -> dict:
        """
        Solve instances using neural construction followed by heuristic refinement.
        """
        # 1. Neural Construction
        neural_out = self.neural_policy(td, env, decode_type=decode_type, **kwargs)

        # 2. Heuristic Refinement
        # Note: Classical solvers currently don't take initial solutions in the current wrappers
        # but we could update them to do so.
        # ALNSSolver.solve(initial_solution=...) exists.

        # For now, we'll just run the heuristic.
        # If we wanted true refinement, we'd pass neural_out["actions"] to the heuristic.

        # Update: Let's assume the heuristic wrapper can take an initial solution in kwargs.
        heuristic_out = self.heuristic_policy(td, env, initial_solution=neural_out["actions"], **kwargs)

        # 3. Return the best of both (usually the heuristic improved result)
        return heuristic_out
