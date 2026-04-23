"""Neural Optimizer (NeuOpt) Policy.

This module implements the `NeuOptPolicy`, which integrates the NeuOpt-specific
Transformer encoder and pairwise decoder for iterative solution improvement.

Attributes:
    NeuOptPolicy: Policy for guided iterative search in combinatorial spaces.
"""

from __future__ import annotations

from typing import Any

from logic.src.models.common.improvement.policy import ImprovementPolicy

from .decoder import NeuOptDecoder
from .encoder import NeuOptEncoder


class NeuOptPolicy(ImprovementPolicy):
    """NeuOpt Policy for iterative improvement.

    Leverages a deep Transformer encoder to capture global problem context
    and a move-selection decoder to identify the most beneficial local
    refinements.

    Attributes:
        encoder (NeuOptEncoder): Deep Transformer encoder for state representation.
        decoder (NeuOptDecoder): Pairwise move selection decoder.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initializes the NeuOpt policy.

        Args:
            embed_dim: dimensionality of the latent representation.
            num_heads: Parallel attention head count.
            num_layers: Depth of the Transformer encoder stacks.
            **kwargs: Extra parameters passed to the improvement base.
        """
        super().__init__(env_name="tsp_kopt", embed_dim=embed_dim)
        self.encoder = NeuOptEncoder(embed_dim, num_heads, num_layers)
        self.decoder = NeuOptDecoder(embed_dim)
