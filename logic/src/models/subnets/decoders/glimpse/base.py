"""
Base context classes for Attention-based Decoders.
"""

from typing import Union

import torch


class AttentionModelFixed:
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached.
    This class allows for efficient indexing of multiple Tensors at once.
    """

    def __init__(
        self,
        node_embeddings: torch.Tensor,
        context_node_projected: torch.Tensor,
        glimpse_key: torch.Tensor,
        glimpse_val: torch.Tensor,
        logit_key: torch.Tensor,
    ):
        self.node_embeddings = node_embeddings
        self.context_node_projected = context_node_projected
        self.glimpse_key = glimpse_key
        self.glimpse_val = glimpse_val
        self.logit_key = logit_key

    def __getitem__(self, key: Union[int, slice, torch.Tensor]) -> "AttentionModelFixed":
        """Allows slicing the fixed context."""
        return AttentionModelFixed(
            self.node_embeddings[key],
            self.context_node_projected[key],
            self.glimpse_key[key],
            self.glimpse_val[key],
            self.logit_key[key],
        )
