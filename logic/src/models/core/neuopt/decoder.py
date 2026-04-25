"""NeuOpt Decoder implementation.

This module implements the `NeuOptDecoder`, which identifies high-quality
local search moves by evaluating node-to-node transition probabilities
conditioned on the global problem context.

Attributes:
    NeuOptDecoder: Pairwise action decoder for guided local search moves.

Example:
    >>> decoder = NeuOptDecoder(embed_dim=128)
    >>> log_p, actions = decoder(td, embeddings, env)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.improvement.decoder import ImprovementDecoder


class NeuOptDecoder(ImprovementDecoder):
    """NeuOpt Decoder for selecting improvement moves.

    Employs a pairwise attention mechanism to compute the similarity between
    all node pairs in a batch, selecting the directed edge or move that promises
    the highest objective improvement.

    Attributes:
        project_q (nn.Linear): Dimensionality projector for query features.
        project_k (nn.Linear): Dimensionality projector for key features.
        scale (float): attention normalization constant.
        seed (int): RNG seed for reproducible move sampling.
        generator (torch.Generator): Local random search generator.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initializes the NeuOpt decoder.

        Args:
            embed_dim: Feature dimension of the input embeddings.
            seed: Random seed for move sampling.
            kwargs: Additional keyword arguments.
        """
        super().__init__(embed_dim=embed_dim)
        self.project_q = nn.Linear(embed_dim, embed_dim)
        self.project_k = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim**-0.5
        self.seed = kwargs.get("seed", seed)
        init_device = kwargs.get("device", "cpu")
        self.generator = torch.Generator(device=init_device).manual_seed(self.seed)

    @property
    def device(self) -> torch.device:
        """Determines the current execution device.

        Returns:
            torch.device: Current placement of solver parameters.
        """
        return next(self.parameters()).device

    def __getstate__(self) -> Dict[str, Any]:
        """Serializes current state for persistence.

        Returns:
            Dict[str, Any]: Attribute map with generator state extracted.
        """
        state = self.__dict__.copy()
        state["generator_state"] = self.generator.get_state()
        state["generator_device"] = str(self.generator.device)
        del state["generator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restores policy parameters and RNG state.

        Args:
            state: Serialized dictionary of solver attributes.
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
        """Predicts a promising neighborhood move (idx1, idx2).

        Args:
            td: TensorDict containing the problem and current solution state.
            embeddings: Node embeddings from the encoder.
            env: The environment defining the move validity and reward.
            kwargs: Additional keyword arguments including "strategy".

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - log_p: log-likelihood of the selected move [B].
                - actions: indices of nodes to modify [B, 2].
        """
        h = embeddings[0] if isinstance(embeddings, tuple) else embeddings
        bs, n, _ = h.shape

        q = self.project_q(h)  # [bs, n, d]
        k = self.project_k(h)  # [bs, n, d]

        # scores: [bs, n, n]
        scores = torch.matmul(q, k.transpose(1, 2)) * self.scale

        # Mask invalid moves (identity diagonal)
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
