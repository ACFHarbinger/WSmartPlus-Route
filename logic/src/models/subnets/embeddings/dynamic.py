"""Dynamic embeddings for step-dependent updates.

This module provides the DynamicEmbedding layer, which aligns with the rl4co
pattern for updating attention keys and values based on time-varying state features.

Attributes:
    DynamicEmbedding: Module for projecting dynamic node features into embedding space.

Example:
    >>> from logic.src.models.subnets.embeddings.dynamic import DynamicEmbedding
    >>> embed = DynamicEmbedding(embed_dim=128)
    >>> d_k, d_v, d_l = embed(td)
"""

from __future__ import annotations

from typing import Tuple

import torch
from tensordict import TensorDict
from torch import nn


class DynamicEmbedding(nn.Module):
    """Dynamic embedding that updates key/values based on current step state.

    Typically used to project dynamic node features (e.g., visited masks, current
    load, or time windows) into components that are added to the static encoder
    context.

    Attributes:
        embed_dim (int): Dimensionality of the target embedding space.
        project_dynamic (nn.Linear): Linear projection for dynamic features.
    """

    def __init__(self, embed_dim: int, dynamic_node_dim: int = 1) -> None:
        """Initializes DynamicEmbedding.

        Args:
            embed_dim: Embedding dimension.
            dynamic_node_dim: Input dimension of dynamic node-level features.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Projection for dynamic node features (e.g., visited mask, remaining waste)
        # We project to 3 * embed_dim to update gl_k, gl_v, logit_k
        self.project_dynamic = nn.Linear(dynamic_node_dim, 3 * embed_dim, bias=False)

    def forward(self, td: TensorDict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes dynamic updates for attention components.

        Checks for 'dynamic_context' or 'visited' in the input state to derive
        node-level updates for glimps keys, values, and selection keys.

        Args:
            td: Current state dictionary.

        Returns:
            Tuple: Updated glimpse keys, glimpse values, and selection logit keys.
        """
        # Default: check for 'visited' mask as a simple dynamic feature
        # visited: [batch, num_loc]
        dynamic_feat = None

        if "dynamic_context" in td.keys():
            # Explicit dynamic context provided by env
            dynamic_feat = td["dynamic_context"]
        elif "visited" in td.keys():
            # Use visited mask as dynamic feature
            # [batch, num_loc] -> [batch, num_loc, 1]
            dynamic_feat = td["visited"].float().unsqueeze(-1)

        if dynamic_feat is None:
            return (
                torch.tensor(0.0, device=td.device),
                torch.tensor(0.0, device=td.device),
                torch.tensor(0.0, device=td.device),
            )

        # Project: [batch, num_loc, 3 * embed_dim]
        out = self.project_dynamic(dynamic_feat)

        # Split
        glimpse_key_dyn, glimpse_val_dyn, logit_key_dyn = out.chunk(3, dim=-1)

        return glimpse_key_dyn, glimpse_val_dyn, logit_key_dyn
