"""
Distribution utilities for node location generation.

Implements various sampling distributions following rl4co patterns:
- Cluster: Gaussian clusters (Bi et al. 2022)
- Mixed: 50% uniform + 50% Gaussian
- Gaussian_Mixture: Configurable GMM (Zhou et al. 2023)
- Mix_Distribution: 33/33/33 mix of uniform/cluster/mixed
- Mix_Multi_Distributions: 11 distribution variants batch-wise
- Gamma: Gamma distribution sampling
- Empirical: Sampling from an empirical dataset
"""

from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from logic.src.pipeline.simulations.bins import Bins


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

        # Generate cluster centers
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
            # Correct logic:
            # We need to construct indices for gathering or scattering
            # mutate_idx shape: (batch, num_loc//2)
            # We take a slice of this for the current cluster
            indices = mutate_idx[:, sum(sizes[:i]) : sum(sizes[: i + 1])]

            means_x = center[:, 2 * i].unsqueeze(1).expand(-1, sizes[i])
            means_y = center[:, 2 * i + 1].unsqueeze(1).expand(-1, sizes[i])

            # Generate new points
            new_points = torch.stack(
                [
                    torch.normal(means_x, self.std),
                    torch.normal(means_y, self.std),
                ],
                dim=2,
            )

            # Scatter back to coords
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
        """Following the setting in Zhang et al. 2022."""
        # Randomly decide how many points each mode gets
        nums = torch.multinomial(
            input=torch.ones(self.num_modes) / self.num_modes,
            num_samples=num_loc,
            replacement=True,
        )

        # Prepare to collect points
        coords = torch.empty((0, 2))

        # Generate points for each mode
        for i in range(self.num_modes):
            num = int((nums == i).sum().item())  # Number of points in this mode
            if num > 0:
                center = torch.rand((1, 2)) * self.cdist
                cov = torch.eye(2)  # Covariance matrix
                nxy = torch.distributions.MultivariateNormal(center.squeeze(), covariance_matrix=cov).sample((num,))
                coords = torch.cat((coords, nxy), dim=0)

        return self._global_min_max_scaling(coords)

    def _generate_gaussian(self, batch_size: int, num_loc: int) -> torch.Tensor:
        """Following the setting in Xin et al. 2022."""
        # Mean and random covariances
        mean = torch.full((batch_size, num_loc, 2), 0.5)
        covs = torch.rand(batch_size)  # Random covariances between 0 and 1

        # Generate the coordinates
        coords = torch.zeros((batch_size, num_loc, 2))
        for i in range(batch_size):
            # Construct covariance matrix for each sample
            cov_matrix = torch.tensor([[1.0, covs[i]], [covs[i], 1.0]])
            m = torch.distributions.MultivariateNormal(mean[i], covariance_matrix=cov_matrix)
            coords[i] = m.sample()

        # Shuffle the coordinates
        indices = torch.randperm(coords.size(0))
        coords = coords[indices]

        return self._batch_normalize_and_center(coords)

    def _global_min_max_scaling(self, coords: torch.Tensor) -> torch.Tensor:
        # Scale the points to [0, 1] using min-max scaling
        coords_min = coords.min(0, keepdim=True).values
        coords_max = coords.max(0, keepdim=True).values
        coords = (coords - coords_min) / (coords_max - coords_min + 1e-8)

        return coords

    def _batch_normalize_and_center(self, coords: torch.Tensor) -> torch.Tensor:
        # Step 1: Compute min and max along each batch
        coords_min = coords.min(dim=1, keepdim=True).values
        coords_max = coords.max(dim=1, keepdim=True).values

        # Step 2: Normalize coordinates to range [0, 1]
        coords = coords - coords_min  # Broadcasting subtracts min value
        range_max = (
            (coords_max - coords_min).max(dim=-1, keepdim=True).values
        )  # The maximum range among both coordinates
        coords = coords / (range_max + 1e-8)  # Divide by the max range

        # Step 3: Center the batch in the middle of the [0, 1] range
        coords = coords + (1 - coords.max(dim=1, keepdim=True).values) / 2  # Centering the batch

        return coords


class Gamma:
    """Gamma distribution."""

    def __init__(self, alpha: Union[float, torch.Tensor] = 2.0, beta: Union[float, torch.Tensor] = 2.0):
        self.alpha = alpha
        self.beta = beta

    def sample(self, size: Tuple[int, ...]) -> torch.Tensor:
        """Sample from Gamma distribution.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc, components))
        """
        # torch.distributions.Gamma(concentration, rate)
        alpha = self.alpha
        beta = self.beta

        # Simple case: scalars or tensors already matching broadcast rules
        m = torch.distributions.Gamma(alpha, beta)
        return m.sample(size)


class Empirical:
    """Sampling from an empirical dataset (e.g. file or Bins object)."""

    def __init__(self, bins: Optional[Bins] = None, dataset_path: Optional[str] = None):
        self.bins = bins
        self.dataset = None
        if bins is None and dataset_path is not None and os.path.exists(dataset_path):
            with open(dataset_path, "rb") as f:
                self.dataset = pickle.load(f)
            # Ensure it's a tensor
            if not isinstance(self.dataset, torch.Tensor):
                try:
                    if isinstance(self.dataset, np.ndarray):
                        self.dataset = torch.from_numpy(self.dataset)
                    elif isinstance(self.dataset, list):
                        self.dataset = torch.tensor(self.dataset)
                except Exception:
                    pass

    def sample(self, size: Tuple[int, ...]) -> torch.Tensor:
        """Sample from empirical dataset.

        Args:
            size: Sampling shape. First dimension is assumed to be batch size if picking whole instances.
        """
        batch_size = size[0]

        # Priority 1: Bins object
        if self.bins is not None:
            # Bins.stochasticFilling returns values in [0, 100]
            # normalized by / 100.0 in data_utils
            vals = self.bins.stochasticFilling(n_samples=batch_size, only_fill=True)
            if isinstance(vals, np.ndarray):
                vals = torch.from_numpy(vals).float()
            return vals / 100.0

        if self.dataset is None:
            # Fallback to uniform if no dataset
            return torch.rand(*size)

        # Priority 2: dataset tensor [N, num_loc, components]
        if isinstance(self.dataset, torch.Tensor):
            # We sample 'batch_size' instances from the dataset
            indices = torch.randint(0, len(self.dataset), (batch_size,))
            out = self.dataset[indices]
            return out

        # Fallback
        return torch.rand(*size)


class Mix_Distribution:
    """33/33/33 mix of Uniform/Cluster/Mixed."""

    def __init__(self, n_cluster: int = 3, n_cluster_mix: int = 1):
        self.Mixed = Mixed(n_cluster_mix=n_cluster_mix)
        self.Cluster = Cluster(n_cluster=n_cluster)

    def sample(self, size: Tuple[int, int, int]) -> torch.Tensor:
        batch_size, num_loc, _ = size
        coords = torch.FloatTensor(batch_size, num_loc, 2).uniform_(0, 1)

        p = torch.rand(batch_size)

        # Mixed: p <= 0.33
        mask = p <= 0.33
        n_mixed = int(mask.sum().item())
        if n_mixed > 0:
            coords[mask] = self.Mixed.sample((n_mixed, num_loc, 2))

        # Cluster: 0.33 < p <= 0.66
        mask = (p > 0.33) & (p <= 0.66)
        n_cluster = int(mask.sum().item())
        if n_cluster > 0:
            coords[mask] = self.Cluster.sample((n_cluster, num_loc, 2))

        return coords


class Mix_Multi_Distributions:
    """11 distribution variants sampled batch-wise."""

    def __init__(self):
        self.distributions: list[tuple[Any, dict[str, Any]]] = [
            (None, {}),  # Uniform
            (Cluster, {"n_cluster": 3}),  # Cluster (3)
            (Mixed, {"n_cluster_mix": 1}),  # Mixed (1)
            (Gaussian_Mixture, {"num_modes": 0, "cdist": 0}),  # Uniform (GMM style)
            (Gaussian_Mixture, {"num_modes": 1, "cdist": 1}),  # Gaussian (GMM style)
        ]
        # Extending to 11 variants
        self.distributions.extend(
            [
                (Cluster, {"n_cluster": 4}),
                (Cluster, {"n_cluster": 5}),
                (Mixed, {"n_cluster_mix": 2}),
                (Mixed, {"n_cluster_mix": 3}),
                (Cluster, {"n_cluster": 6}),
                (Mixed, {"n_cluster_mix": 4}),
            ]
        )

    def sample(self, size: Tuple[int, int, int]) -> torch.Tensor:
        batch_size, num_loc, _ = size
        coords = torch.zeros(batch_size, num_loc, 2)

        # Assign each sample in batch to a random distribution type
        dist_indices = torch.randint(0, len(self.distributions), (batch_size,))

        # Group by distribution to batch calls
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


# Registry
DISTRIBUTION_REGISTRY: dict[str, Callable[..., Any]] = {
    "uniform": lambda: None,  # Use default torch.rand
    "cluster": Cluster,
    "mixed": Mixed,
    "gaussian_mixture": Gaussian_Mixture,
    "gamma": Gamma,
    "empirical": Empirical,
    "mix_distribution": Mix_Distribution,
    "mix_multi": Mix_Multi_Distributions,
}
