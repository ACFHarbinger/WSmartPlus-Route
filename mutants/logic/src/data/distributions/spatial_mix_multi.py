from typing import Tuple

import torch

from .spatial_cluster import Cluster
from .spatial_gaussian_mixture import Gaussian_Mixture
from .spatial_mixed import Mixed


class Mix_Multi_Distributions:
    """Batch-wise sampling from multiple distribution variants."""

    def __init__(self):
        self.distributions = [
            (None, {}),
            (Cluster, {"n_cluster": 3}),
            (Mixed, {"n_cluster_mix": 1}),
            (Gaussian_Mixture, {"num_modes": 0, "cdist": 0}),
            (Gaussian_Mixture, {"num_modes": 1, "cdist": 1}),
            (Cluster, {"n_cluster": 4}),
            (Cluster, {"n_cluster": 5}),
            (Mixed, {"n_cluster_mix": 2}),
            (Mixed, {"n_cluster_mix": 3}),
            (Cluster, {"n_cluster": 6}),
            (Mixed, {"n_cluster_mix": 4}),
        ]

    def sample(self, size: Tuple[int, int, int]) -> torch.Tensor:
        batch_size, num_loc, _ = size
        coords = torch.zeros(batch_size, num_loc, 2)
        dist_indices = torch.randint(0, len(self.distributions), (batch_size,))

        for i, (cls, kwargs) in enumerate(self.distributions):
            mask = dist_indices == i
            n_samples = int(mask.sum().item())
            if n_samples == 0:
                continue

            if cls is None:
                coords[mask] = torch.rand(n_samples, num_loc, 2)
            else:
                dist = cls(**kwargs)
                coords[mask] = dist.sample((n_samples, num_loc, 2))

        return coords.clamp_(0, 1)
