"""DACT Policy architecture.

This module implements the `DACTPolicy`, which integrates the Dual Aspect
Collaborative Transformer architecture (Ma et al. 2021) into an iterative
improvement framework.

Attributes:
    DACTPolicy: Collaborative Transformer policy for routing improvement.

Example:
    >>> from logic.src.models.core.dact.policy import DACTPolicy
    >>> policy = DACTPolicy(env_name="tsp_kopt", embed_dim=128)
"""

from __future__ import annotations

from typing import Any

from logic.src.models.common.improvement.policy import ImprovementPolicy

from .decoder import DACTDecoder
from .encoder import DACTEncoder


class DACTPolicy(ImprovementPolicy):
    """DACT (Dual Aspect Collaborative Transformer) Policy.

    Specializes in iterative improvements by simultaneously processing spatial
    node attributes and sequence-dependent solution features. It extends
    `ImprovementPolicy` with a DACT-specific encoder and decoder.

    Attributes:
        encoder (DACTEncoder): Hierarchical Transformer for problem context.
        decoder (DACTDecoder): Attention-based operator selector.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        env_name: str = "tsp_kopt",
        **kwargs: Any,
    ) -> None:
        """Initializes the DACT policy.

        Args:
            embed_dim: Dimensionality of latent embeddings.
            num_layers: Number of transformer layers in the encoder.
            num_heads: Number of attention heads.
            env_name: Name of the environment identifier.
            kwargs: Additional keyword arguments.
        """
        encoder = DACTEncoder(embed_dim, num_layers, num_heads, **kwargs)
        decoder = DACTDecoder(embed_dim, num_heads, **kwargs)

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            embed_dim=embed_dim,
            **kwargs,
        )
