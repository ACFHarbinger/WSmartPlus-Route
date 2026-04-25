"""DACT Decoder implementation.

This module implements the `DACTDecoder`, which selects node pairs (i, j) to
perform local improvement moves (e.g., 2-opt) using a pairwise attention
mechanism.

Attributes:
    DACTDecoder: Pairwise action decoder for iterative improvement.

Example:
    >>> from logic.src.models.core.dact.decoder import DACTDecoder
    >>> decoder = DACTDecoder(embed_dim=128)
    >>> log_p, actions = decoder(td, h, my_env)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.improvement.policy import ImprovementDecoder


class DACTDecoder(ImprovementDecoder):
    """DACT Pairwise Action Decoder.

    Processes node embeddings to predict the optimal pair of indices for iterative
    improvement moves. It uses a cross-attention score matrix to evaluate all
    possible node combinations (i, j).

    Attributes:
        num_heads (int): count of attention heads.
        seed (int): Random seed for sampling stochasticity.
        generator (torch.Generator): PRNG instance for reproducible sampling.
        project_q (nn.Linear): Transformation for query nodes.
        project_k (nn.Linear): Transformation for key nodes.
        scale (float): Attention score normalization factor.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initializes the DACT decoder.

        Args:
            embed_dim: Dimensionality of latent embeddings.
            num_heads: Number of attention heads.
            seed: Random seed for action sampling.
            kwargs: Additional keyword arguments.
        """
        super().__init__(embed_dim)
        self.num_heads = num_heads
        self.seed = seed
        init_device = kwargs.get("device", "cpu")
        self.generator = torch.Generator(device=init_device).manual_seed(self.seed)

        self.project_q = nn.Linear(embed_dim, embed_dim)
        self.project_k = nn.Linear(embed_dim, embed_dim)

        self.scale = 1.0 / (embed_dim // num_heads) ** 0.5

    @property
    def device(self) -> torch.device:
        """Determines the current hardware placement of the model.

        Returns:
            torch.device: Current placement of the model parameters.
        """
        return next(self.parameters()).device

    def __getstate__(self) -> Dict[str, Any]:
        """Serializes the state, handling non-picklable components.

        Returns:
            Dict[str, Any]: Model state dictionary with generator metadata.
        """
        state = self.__dict__.copy()
        state["generator_state"] = self.generator.get_state()
        state["generator_device"] = str(self.generator.device)
        del state["generator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restores the model state including the RNG generator.

        Args:
            state: Serialized attribute map.
        """
        gen_state = state.pop("generator_state")
        gen_device = state.pop("generator_device")
        self.__dict__.update(state)
        self.generator = torch.Generator(device=gen_device)
        self.generator.set_state(gen_state)

    def forward(
        self,
        td: TensorDict,
        embeddings: torch.Tensor | Tuple[torch.Tensor, ...],
        env: RL4COEnvBase,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts a node pair for an improvement operator.

        Args:
            td: TensorDict containing problem instance data.
            embeddings: Contextual node features or feature tuple.
            env: Environment managing problem physics.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - log_p: Log-probability of the selected pair [B].
                - actions: Integer indices [i, j] for the move [B, 2].
        """
        h = embeddings[0] if isinstance(embeddings, tuple) else embeddings
        bs, n, _ = h.shape

        # Pairwise attention scoring
        q = self.project_q(h)
        k = self.project_k(h)
        scores = torch.matmul(q, k.transpose(1, 2)) * self.scale

        # Mask identity moves (i == j)
        mask = torch.eye(n, device=h.device).bool().unsqueeze(0).expand(bs, -1, -1)
        scores = scores.masked_fill(mask, float("-inf"))

        logits = scores.view(bs, -1)

        # Action selection
        strategy = kwargs.get("strategy", "greedy")
        if strategy == "greedy":
            action_indices = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            action_indices = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)

        # Coordinate conversion
        idx_i = action_indices // n
        idx_j = action_indices % n
        actions = torch.stack([idx_i, idx_j], dim=-1)

        log_p = F.log_softmax(logits, dim=-1).gather(1, action_indices.unsqueeze(-1)).squeeze(-1)

        return log_p, actions
