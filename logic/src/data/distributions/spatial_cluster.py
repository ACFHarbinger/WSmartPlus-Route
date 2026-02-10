"""spatial_cluster.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import spatial_cluster
"""

from typing import Tuple

import torch


class Cluster:
    """Multiple Gaussian distributed clusters."""

    def __init__(self, n_cluster: int = 3):
        """Initialize Class.

        Args:
            n_cluster (int): Description of n_cluster.
        """
        self.n_cluster = n_cluster
        self.lower, self.upper = 0.2, 0.8
        self.std = 0.07

    def sample(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """Sample clustered locations.

        Args:
            size: (batch_size, num_loc, 2)

        Returns:
            Tensor of shape (batch_size, num_loc, 2)
        """
        batch_size, num_loc, _ = size

        center = self.lower + (self.upper - self.lower) * torch.rand(batch_size, self.n_cluster * 2)

        coords = torch.zeros(batch_size, num_loc, 2)
        cluster_sizes = [num_loc // self.n_cluster] * self.n_cluster
        for i in range(num_loc % self.n_cluster):
            cluster_sizes[i] += 1

        current_index = 0
        for i in range(self.n_cluster):
            means = center[:, i * 2 : (i + 1) * 2]
            stds = torch.full((batch_size, 2), self.std)
            points = torch.normal(
                means.unsqueeze(1).expand(-1, cluster_sizes[i], -1),
                stds.unsqueeze(1).expand(-1, cluster_sizes[i], -1),
            )
            coords[:, current_index : (current_index + cluster_sizes[i]), :] = points
            current_index += cluster_sizes[i]

        return coords.clamp_(0, 1)
