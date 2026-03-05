"""spatial_mix.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import spatial_mix
"""

from typing import Optional, Tuple

import numpy as np
import torch

from .base import BaseDistribution
from .spatial_cluster import Cluster
from .spatial_mixed import Mixed


class MixDistribution(BaseDistribution):
    """33/33/33 mix of Uniform/Cluster/Mixed."""

    def __init__(self, n_cluster: int = 3, n_cluster_mix: int = 1):
        """Initialize Class.

        Args:
            n_cluster (int): Description of n_cluster.
            n_cluster_mix (int): Description of n_cluster_mix.
        """
        self.Mixed = Mixed(n_cluster_mix=n_cluster_mix)
        self.Cluster = Cluster(n_cluster=n_cluster)

    def _sample_tensor(self, size: Tuple[int, int, int], generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample.

        Args:
            size (Tuple[int, int, int]): Description of size.
            generator (Optional[torch.Generator], optional): Description of generator.

        Returns:
            Any: Description of return value.
        """
        if generator is None:
            generator = torch.Generator().manual_seed(42)

        batch_size, num_loc, _ = size
        coords = torch.FloatTensor(batch_size, num_loc, 2).uniform_(0, 1)

        p = torch.rand(batch_size, generator=generator)

        mask = p <= 0.33
        n_mixed = int(mask.sum().item())
        if n_mixed > 0:
            coords[mask] = self.Mixed._sample_tensor((n_mixed, num_loc, 2), generator=generator)

        mask = (p > 0.33) & (p <= 0.66)
        n_cluster = int(mask.sum().item())
        if n_cluster > 0:
            coords[mask] = self.Cluster._sample_tensor((n_cluster, num_loc, 2), generator=generator)

        return coords

    def _sample_array(self, size: Tuple[int, int, int], rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """NumPy version of the probabilistic spatial sampler."""
        if rng is None:
            rng = np.random.RandomState(42)

        batch_size, num_loc, _ = size

        # 1. Initialize coordinates with uniform distribution [0, 1)
        coords = rng.rand(batch_size, num_loc, 2)

        # 2. Sample probabilities for the distribution mixture
        p = rng.rand(batch_size)

        # 3. Handle 'Mixed' distribution (p <= 0.33)
        mask_mixed = p <= 0.33
        n_mixed = np.sum(mask_mixed)
        if n_mixed > 0:
            # Strategy Pattern: call the array version of the Mixed sampler
            coords[mask_mixed] = self.Mixed._sample_array((n_mixed, num_loc, 2), rng=rng)

        # 4. Handle 'Cluster' distribution (0.33 < p <= 0.66)
        mask_cluster = (p > 0.33) & (p <= 0.66)
        n_cluster = np.sum(mask_cluster)
        if n_cluster > 0:
            # Strategy Pattern: call the array version of the Cluster sampler
            coords[mask_cluster] = self.Cluster._sample_array((n_cluster, num_loc, 2), rng=rng)

        # Note: (p > 0.66) remains the initial uniform distribution (coords)
        return coords
