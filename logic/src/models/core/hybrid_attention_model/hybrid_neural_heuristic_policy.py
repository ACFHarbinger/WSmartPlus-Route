"""Neural Heuristic Hybrid: Constructive-Refinement Policy.

This module implements a hybrid execution flow where a neural network generates
an initial solution (constructive phase), and a classic meta-heuristic
(refinement phase) iterates upon it to reach local optimality.

Attributes:
    NeuralHeuristicHybrid: Unified policy for construction and refinement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Union

from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.autoregressive.policy import AutoregressivePolicy

if TYPE_CHECKING:
    from logic.src.models.policies.alns import VectorizedALNS
    from logic.src.models.policies.hgs import VectorizedHGS


class NeuralHeuristicHybrid(AutoregressivePolicy):
    """Hybrid policy combining deep learning construction and OR-based refinement.

    Attributes:
        neural_policy (AutoregressivePolicy): The model used for Stage 1.
        heuristic_policy (Union[VectorizedALNS, VectorizedHGS]): Classic solver
            used for Stage 2 refinement.
    """

    def __init__(
        self,
        neural_policy: AutoregressivePolicy,
        heuristic_policy: Union[VectorizedALNS, VectorizedHGS],
        **kwargs: Any,
    ) -> None:
        """Initializes the hybrid policy.

        Args:
            neural_policy: Pre-trained constructive model.
            heuristic_policy: Vectorized meta-heuristic instance.
            **kwargs: Extra parameters for base policy.
        """
        super().__init__(env_name=neural_policy.env_name, **kwargs)
        self.neural_policy = neural_policy
        self.heuristic_policy = heuristic_policy

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        strategy: str = "greedy",
        num_starts: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Solves problem instances using the hybrid execution flow.

        Args:
            td: problem state.
            env: dynamics and rewards.
            strategy: constructive decoding mode.
            num_starts: multi-start count for construction.
            **kwargs: extra parameters passed to both components.

        Returns:
            Dict[str, Any]: Refinement results including 'actions' and 'reward'.
        """
        # Phase 1: Contextual Neural Construction
        neural_out = self.neural_policy(td, env, strategy=strategy, **kwargs)

        # Phase 2: Heuristic Refinement
        # The heuristic solver starts from the neural-generated sequences
        heuristic_out = self.heuristic_policy(td, env, initial_solution=neural_out["actions"], **kwargs)

        return heuristic_out
