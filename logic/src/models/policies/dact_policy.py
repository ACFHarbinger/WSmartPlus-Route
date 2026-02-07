from __future__ import annotations

from .common.improvement import ImprovementPolicy
from .dact_decoder import DACTDecoder
from .dact_encoder import DACTEncoder


class DACTPolicy(ImprovementPolicy):
    """
    DACT Policy: Collaborative Transformer for iterative improvement.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        env_name: str = "tsp_kopt",
        **kwargs,
    ):
        """Initialize DACTPolicy."""
        encoder = DACTEncoder(embed_dim, num_layers, num_heads, **kwargs)
        decoder = DACTDecoder(embed_dim, num_heads, **kwargs)

        super().__init__(encoder=encoder, decoder=decoder, env_name=env_name, embed_dim=embed_dim, **kwargs)
