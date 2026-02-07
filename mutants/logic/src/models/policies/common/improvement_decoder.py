from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch
import torch.nn as nn
from logic.src.envs.base import RL4COEnvBase
from tensordict import TensorDict


class ImprovementDecoder(nn.Module, ABC):
    """
    Base class for improvement decoders.

    Improvement decoders predict moves or operators to apply to the current solution.
    """

    def __init__(self, embed_dim: int = 128, **kwargs):
        """Initialize ImprovementDecoder."""
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        embeddings: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        env: RL4COEnvBase,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict improvement moves.

        Args:
            td: TensorDict containing current state.
            embeddings: Encoded state and solution.
            env: Environment for state transitions.

        Returns:
            Tuple of (log_likelihood, actions/moves).
        """
        raise NotImplementedError
