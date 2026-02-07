"""
Spatial sampling distributions for node location generation.
"""

from typing import Tuple

import torch


class Cluster:
    """Multiple Gaussian distributed clusters."""

    def __init__(self, n_cluster: int = 3):
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


class Mixed:
    """50% uniform + 50% Gaussian clusters."""

    def __init__(self, n_cluster_mix: int = 1):
        self.n_cluster_mix = n_cluster_mix
        self.lower, self.upper = 0.2, 0.8
        self.std = 0.07

    def sample(self, size: Tuple[int, int, int]) -> torch.Tensor:
        batch_size, num_loc, _ = size

        center = self.lower + (self.upper - self.lower) * torch.rand(batch_size, self.n_cluster_mix * 2)

        coords = torch.FloatTensor(batch_size, num_loc, 2).uniform_(0, 1)
        mutate_idx = torch.stack([torch.randperm(num_loc)[: num_loc // 2] for _ in range(batch_size)])

        segment_size = num_loc // (2 * self.n_cluster_mix)
        remaining = num_loc // 2 - segment_size * (self.n_cluster_mix - 1)
        sizes = [segment_size] * (self.n_cluster_mix - 1) + [remaining]

        for i in range(self.n_cluster_mix):
            indices = mutate_idx[:, sum(sizes[:i]) : sum(sizes[: i + 1])]

            means_x = center[:, 2 * i].unsqueeze(1).expand(-1, sizes[i])
            means_y = center[:, 2 * i + 1].unsqueeze(1).expand(-1, sizes[i])

            new_points = torch.stack(
                [
                    torch.normal(means_x, self.std),
                    torch.normal(means_y, self.std),
                ],
                dim=2,
            )

            coords.scatter_(
                1,
                indices.unsqueeze(-1).expand(-1, -1, 2),
                new_points,
            )

        return coords.clamp_(0, 1)


class Gaussian_Mixture:
    """Configurable Gaussian Mixture Model."""

    def __init__(self, num_modes: int = 0, cdist: int = 0):
        self.num_modes = num_modes
        self.cdist = cdist

    def sample(self, size: Tuple[int, int, int]) -> torch.Tensor:
        batch_size, num_loc, _ = size

        if self.num_modes == 0:
            return torch.rand((batch_size, num_loc, 2))
        elif self.num_modes == 1 and self.cdist == 1:
            return self._generate_gaussian(batch_size, num_loc)
        else:
            return torch.stack([self._generate_gaussian_mixture(num_loc) for _ in range(batch_size)])

    def _generate_gaussian_mixture(self, num_loc: int) -> torch.Tensor:
        nums = torch.multinomial(
            input=torch.ones(self.num_modes) / self.num_modes,
            num_samples=num_loc,
            replacement=True,
        )

        coords = torch.empty((0, 2))

        for i in range(self.num_modes):
            num = int((nums == i).sum().item())
            if num > 0:
                center = torch.rand((1, 2)) * self.cdist
                cov = torch.eye(2)
                nxy = torch.distributions.MultivariateNormal(center.squeeze(), covariance_matrix=cov).sample(
                    torch.Size([num])
                )
                coords = torch.cat((coords, nxy), dim=0)

        return self._global_min_max_scaling(coords)

    def _generate_gaussian(self, batch_size: int, num_loc: int) -> torch.Tensor:
        mean = torch.full((batch_size, num_loc, 2), 0.5)
        covs = torch.rand(batch_size)

        coords = torch.zeros((batch_size, num_loc, 2))
        for i in range(batch_size):
            cov_matrix = torch.tensor([[1.0, covs[i]], [covs[i], 1.0]])
            m = torch.distributions.MultivariateNormal(mean[i], covariance_matrix=cov_matrix)
            coords[i] = m.sample()

        indices = torch.randperm(coords.size(0))
        coords = coords[indices]

        return self._batch_normalize_and_center(coords)

    def _global_min_max_scaling(self, coords: torch.Tensor) -> torch.Tensor:
        coords_min = coords.min(0, keepdim=True).values
        coords_max = coords.max(0, keepdim=True).values
        coords = (coords - coords_min) / (coords_max - coords_min + 1e-8)

        return coords

    def _batch_normalize_and_center(self, coords: torch.Tensor) -> torch.Tensor:
        coords_min = coords.min(dim=1, keepdim=True).values
        coords_max = coords.max(dim=1, keepdim=True).values

        coords = coords - coords_min
        range_max = (coords_max - coords_min).max(dim=-1, keepdim=True).values
        coords = coords / (range_max + 1e-8)

        coords = coords + (1 - coords.max(dim=1, keepdim=True).values) / 2

        return coords


class Mix_Distribution:
    """33/33/33 mix of Uniform/Cluster/Mixed."""

    def __init__(self, n_cluster: int = 3, n_cluster_mix: int = 1):
        self.Mixed = Mixed(n_cluster_mix=n_cluster_mix)
        self.Cluster = Cluster(n_cluster=n_cluster)

    def sample(self, size: Tuple[int, int, int]) -> torch.Tensor:
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
