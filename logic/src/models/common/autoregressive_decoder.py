from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch
import torch.nn as nn
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase


class AutoregressiveDecoder(nn.Module, ABC):
    """
    Base class for autoregressive decoders.

    AR decoders select actions sequentially until a termination condition is met.
    """

    def __init__(self, embed_dim: int = 128, **kwargs):
        """Initialize AutoregressiveDecoder."""
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
        Construct solutions step-by-step.

        Args:
            td: TensorDict containing problem instance.
            embeddings: Encoded state from the encoder.
            env: Environment for state transitions.

        Returns:
            Tuple of (log_likelihood, actions).
        """
        raise NotImplementedError
