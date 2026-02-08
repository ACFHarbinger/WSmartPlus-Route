"""spatial_mix.py module.

    Attributes:
        MODULE_VAR (Type): Description of module level variable.

    Example:
        >>> import spatial_mix
    """
from typing import Tuple

import torch

from .spatial_cluster import Cluster
from .spatial_mixed import Mixed


class Mix_Distribution:
    """33/33/33 mix of Uniform/Cluster/Mixed."""

    def __init__(self, n_cluster: int = 3, n_cluster_mix: int = 1):
        """Initialize Class.

        Args:
            n_cluster (int): Description of n_cluster.
            n_cluster_mix (int): Description of n_cluster_mix.
        """
        self.Mixed = Mixed(n_cluster_mix=n_cluster_mix)
        self.Cluster = Cluster(n_cluster=n_cluster)

    def sample(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """Sample.

        Args:
            size (Tuple[int, int, int]): Description of size.

        Returns:
            Any: Description of return value.
        """
        batch_size, num_loc, _ = size
        coords = torch.FloatTensor(batch_size, num_loc, 2).uniform_(0, 1)

        p = torch.rand(batch_size)

        mask = p <= 0.33
        n_mixed = int(mask.sum().item())
        if n_mixed > 0:
            coords[mask] = self.Mixed.sample((n_mixed, num_loc, 2))

        mask = (p > 0.33) & (p <= 0.66)
        n_cluster = int(mask.sum().item())
        if n_cluster > 0:
            coords[mask] = self.Cluster.sample((n_cluster, num_loc, 2))

        return coords
