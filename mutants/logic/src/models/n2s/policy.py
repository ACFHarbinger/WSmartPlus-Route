from __future__ import annotations

from .common.improvement import ImprovementPolicy
from .n2s_decoder import N2SDecoder
from .n2s_encoder import N2SEncoder


class N2SPolicy(ImprovementPolicy):
    """
    N2S Policy: Iterative neighborhood search.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        k_neighbors: int = 20,
        **kwargs,
    ):
        super().__init__(env_name="tsp_kopt", embed_dim=embed_dim)
        self.encoder = N2SEncoder(embed_dim, num_heads, k_neighbors)
        self.decoder = N2SDecoder(embed_dim)
