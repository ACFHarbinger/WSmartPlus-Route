"""Neural Neighborhood Search (N2S) Policy.

This module implements the `N2SPolicy`, which integrates the N2S-specific
encoder and decoder for iterative improvement of combinatorial solutions.

Attributes:
    N2SPolicy: Collaborative Transformer policy for neighborhood search.
"""

from __future__ import annotations

from typing import Any

from logic.src.models.common.improvement.policy import ImprovementPolicy

from .decoder import N2SDecoder
from .encoder import N2SEncoder


class N2SPolicy(ImprovementPolicy):
    """N2S Policy for iterative routing improvement.

    Leverages a spatial-aware encoder and a move-selection decoder to
    iteratively refine a solution by exploring and selecting local
    neighborhood moves.

    Attributes:
        encoder (N2SEncoder): Transformer encoder for problem and solution state.
        decoder (N2SDecoder): Pairwise node-selection decoder.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        k_neighbors: int = 20,
        **kwargs: Any,
    ) -> None:
        """Initializes the N2S policy.

        Args:
            embed_dim: Width of the latent feature space.
            num_heads: count of attention heads for the internal subnets.
            k_neighbors: number of local neighbors to consider per node.
            **kwargs: Extra parameters passed to the improvement base.
        """
        super().__init__(env_name="tsp_kopt", embed_dim=embed_dim)
        self.encoder = N2SEncoder(embed_dim, num_heads, k_neighbors)
        self.decoder = N2SDecoder(embed_dim)
