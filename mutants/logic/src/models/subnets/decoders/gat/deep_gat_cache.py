"""Context for GAT decoder."""

from __future__ import annotations

import typing

import torch


class DeepAttentionModelFixed(typing.NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """

    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor

    def __getitem__(self, key):
        """Get item from DeepAttentionModelFixed."""
        if torch.is_tensor(key) or isinstance(key, slice):
            return DeepAttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
            )
        return DeepAttentionModelFixed(
            node_embeddings=self.node_embeddings[key].unsqueeze(0),
            context_node_projected=self.context_node_projected[key].unsqueeze(0),
        )
