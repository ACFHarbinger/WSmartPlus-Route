"""DeepACO Decoder: Ant colony solution construction from heatmaps.

This module provides the ACO-based decoder which uses pheromone-guided construction
directed by GNN-predicted heatmaps.

Attributes:
    ACODecoder: Ant Colony Optimization decoder for solution construction.

Example:
    >>> from logic.src.models.subnets.decoders.deepaco import ACODecoder
    >>> decoder = ACODecoder(n_ants=20, alpha=1.0, beta=2.0)
"""

from .decoder import ACODecoder as ACODecoder

__all__ = [
    "ACODecoder",
]
