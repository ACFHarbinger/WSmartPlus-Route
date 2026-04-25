"""N2S Decoder implementation.

This module implements the `N2SDecoder`, which selects node pairs for local
neighborhood moves based on the learned node embeddings and spatial
constraints.

Attributes:
    N2SDecoder: Pairwise action decoder for iterative neighborhood search.

Example:
    >>> decoder = N2SDecoder(embed_dim=128)
    >>> log_p, actions = decoder(td, h, env)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.improvement.decoder import ImprovementDecoder


class N2SDecoder(ImprovementDecoder):
    """N2S Decoder for selecting neighborhood moves.

    Uses a pairwise attention mechanism to evaluate all valid node combinations
    within a batch, selecting the move that most likely leads to a solution
    improvement.

    Attributes:
        project_q (nn.Linear): Transformation for query nodes.
        project_k (nn.Linear): Transformation for key nodes.
        scale (float): Attention normalization factor.
        seed (int): RNG seed for sampling.
        generator (torch.Generator): Local random search generator.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initializes the N2S decoder.

        Args:
            embed_dim: Dimensionality of the node features.
            seed: Random seed for action sampling.
            kwargs: Additional keyword arguments.
        """
        super().__init__(embed_dim=embed_dim)
        self.project_q = nn.Linear(embed_dim, embed_dim)
        self.project_k = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim**-0.5
        self.seed = seed
        init_device = kwargs.get("device", "cpu")
        self.generator = torch.Generator(device=init_device).manual_seed(self.seed)

    @property
    def device(self) -> torch.device:
        """Determines the current hardware placement of parameters.

        Returns:
            torch.device: The active device (CPU/CUDA).
        """
        return next(self.parameters()).device

    def __getstate__(self) -> Dict[str, Any]:
        """Serializes the state, handling non-picklable components.

        Returns:
            Dict[str, Any]: Attribute map with generator state recorded.
        """
        state = self.__dict__.copy()
        state["generator_state"] = self.generator.get_state()
        state["generator_device"] = str(self.generator.device)
        del state["generator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restores policy state including the RNG generator.

        Args:
            state: Serialized dictionary of attributes.
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
        """Predicts an improvement move (node pair i, j).

        Args:
            td: TensorDict containing problem state.
            embeddings: Encoded node features [B, N, D].
            env: Environment managing the problem physics.
            kwargs: Additional keyword arguments including decoding strategy.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - log_p: log-likelihood of the selected move [B].
                - actions: pair of node indices [B, 2].
        """
        h = embeddings[0] if isinstance(embeddings, tuple) else embeddings
        bs, n, _ = h.shape

        q = self.project_q(h)  # [bs, n, d]
        k = self.project_k(h)  # [bs, n, d]

        # scores: [bs, n, n]
        scores = torch.matmul(q, k.transpose(1, 2)) * self.scale

        # Mask identity moves (i == j)
        mask = torch.eye(n, device=h.device).bool().unsqueeze(0).expand(bs, -1, -1)
        scores.masked_fill_(mask, float("-inf"))

        logits = scores.view(bs, -1)

        strategy = kwargs.get("strategy", "greedy")
        if strategy == "greedy":
            action_indices = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            action_indices = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)

        # Convert flat index back to (i, j)
        idx1 = torch.div(action_indices, n, rounding_mode="floor")
        idx2 = action_indices % n
        actions = torch.stack([idx1, idx2], dim=-1)

        log_p = F.log_softmax(logits, dim=-1).gather(1, action_indices.unsqueeze(-1)).squeeze(-1)

        return log_p, actions
