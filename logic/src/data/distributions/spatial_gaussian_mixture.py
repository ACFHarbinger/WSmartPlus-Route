"""spatial_gaussian_mixture.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import spatial_gaussian_mixture
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch

from .base import BaseDistribution


class GaussianMixture(BaseDistribution):
    """Configurable Gaussian Mixture Model."""

    def __init__(self, num_modes: int = 0, cdist: int = 0):
        """Initialize Class.

        Args:
            num_modes (int): Description of num_modes.
            cdist (int): Description of cdist.
        """
        self.num_modes = num_modes
        self.cdist = cdist

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
        if self.num_modes == 0:
            return torch.rand((batch_size, num_loc, 2), generator=generator)
        elif self.num_modes == 1 and self.cdist == 1:
            return self._generate_gaussian(batch_size, num_loc, generator=generator)
        else:
            return torch.stack(
                [self._generate_gaussian_mixture(num_loc, generator=generator) for _ in range(batch_size)]
            )

    def _sample_array(self, size: Tuple[int, int, int], rng: Optional[np.random.default_rng] = None) -> np.ndarray:
        """NumPy version of the Gaussian/Mixture spatial sampler."""
        if rng is None:
            # Maintaining the same default seed for reproducibility
            rng = np.random.default_rng(42)

        batch_size, num_loc, _ = size

        # Case 1: Simple Uniform Distribution
        if self.num_modes == 0:
            return rng.rand(batch_size, num_loc, 2)

        # Case 2: Single Mode Gaussian (Vectorized)
        elif self.num_modes == 1 and self.cdist == 1:
            # Ensure _generate_gaussian_array is implemented using rng
            return self._generate_gaussian_array(batch_size, num_loc, rng=rng)

        # Case 3: Gaussian Mixture (Iterative/Batch)
        else:
            # We use a list comprehension and stack, similar to the torch.stack approach
            # Note: If _generate_gaussian_mixture can be vectorized for batches,
            # that would be more efficient than this loop.
            samples = [self._generate_gaussian_mixture_array(num_loc, rng=rng) for _ in range(batch_size)]
        return np.stack(samples)

    def _generate_gaussian_mixture(self, num_loc: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """generate gaussian mixture.

        Args:
            num_loc (int): Description of num_loc.
            generator (Optional[torch.Generator], optional): Description of generator.

        Returns:
            Any: Description of return value.
        """
        nums = torch.multinomial(
            input=torch.ones(self.num_modes) / self.num_modes,
            num_samples=num_loc,
            replacement=True,
            generator=generator,
        )

        coords = torch.empty((0, 2))

        for i in range(self.num_modes):
            num = int((nums == i).sum().item())
            if num > 0:
                center = torch.rand((1, 2), generator=generator) * self.cdist
                cov = torch.eye(2)
                nxy = torch.distributions.MultivariateNormal(center.squeeze(), covariance_matrix=cov).sample(
                    torch.Size([num])
                )
                coords = torch.cat((coords, nxy), dim=0)

        return self._global_min_max_scaling(coords)

    def _generate_gaussian(
        self, batch_size: int, num_loc: int, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """generate gaussian.

        Args:
            batch_size (int): Description of batch_size.
            num_loc (int): Description of num_loc.
            generator (Optional[torch.Generator], optional): Description of generator.

        Returns:
            Any: Description of return value.
        """
        mean = torch.full((batch_size, num_loc, 2), 0.5)
        covs = torch.rand(batch_size, generator=generator)

        coords = torch.zeros((batch_size, num_loc, 2))
        for i in range(batch_size):
            cov_matrix = torch.tensor([[1.0, covs[i]], [covs[i], 1.0]])
            m = torch.distributions.MultivariateNormal(mean[i], covariance_matrix=cov_matrix)
            coords[i] = m.sample()

        indices = torch.randperm(coords.size(0), generator=generator)
        coords = coords[indices]

        return self._batch_normalize_and_center(coords)

    def _generate_gaussian_mixture_array(self, num_loc: int, rng: np.random.default_rng) -> np.ndarray:
        # 1. Sample which mode each point belongs to (replacing multinomial)
        probs = np.ones(self.num_modes) / self.num_modes
        nums = rng.choice(self.num_modes, size=num_loc, p=probs)

        coords_list = []

        for i in range(self.num_modes):
            num = np.sum(nums == i)
            if num > 0:
                # Random center scaled by cdist
                center = rng.rand(2) * self.cdist
                cov = np.eye(2)

                # Sample from Multivariate Normal
                nxy = rng.multivariate_normal(mean=center, cov=cov, size=num)
                coords_list.append(nxy)

        # Concatenate all sampled clusters
        coords = np.vstack(coords_list)

        return self._global_min_max_scaling(coords)

    def _generate_gaussian_array(self, batch_size: int, num_loc: int, rng: np.random.default_rng) -> np.ndarray:
        coords = np.zeros((batch_size, num_loc, 2))
        cov_values = rng.rand(batch_size)

        # In NumPy, we still iterate over the batch for Multivariate Normal
        # as its built-in multivariate_normal doesn't broadcast different cov matrices easily
        for i in range(batch_size):
            mean = np.array([0.5, 0.5])
            # Construct cov matrix: [[1.0, rho], [rho, 1.0]]
            cov_matrix = np.array([[1.0, cov_values[i]], [cov_values[i], 1.0]])

            # Sample directly into the coordinate slice
            coords[i] = rng.multivariate_normal(mean=mean, cov=cov_matrix, size=num_loc)

        # Shuffling the batch (equivalent to randperm)
        indices = rng.permutation(batch_size)
        coords = coords[indices]

        return self._batch_normalize_and_center_array(coords)

    def _global_min_max_scaling(self, coords: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """global min max scaling.

        Args:
            coords (Union[np.ndarray, torch.Tensor]): Description of coords.

        Returns:
            Any: Description of return value.
        """
        if isinstance(coords, torch.Tensor):
            coords_min = coords.min(dim=0, keepdim=True).values
            coords_max = coords.max(dim=0, keepdim=True).values
        else:
            coords_min = coords.min(axis=0, keepdims=True)
            coords_max = coords.max(axis=0, keepdims=True)

        coords = (coords - coords_min) / (coords_max - coords_min + 1e-8)
        return coords

    def _batch_normalize_and_center(self, coords: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """batch normalize and center.

        Args:
            coords (Union[np.ndarray, torch.Tensor]): Description of coords.

        Returns:
            Any: Description of return value.
        """
        if isinstance(coords, torch.Tensor):
            coords_min = coords.min(dim=1, keepdim=True).values
            coords_max = coords.max(dim=1, keepdim=True).values
        else:
            coords_min = coords.min(axis=1, keepdims=True)
            coords_max = coords.max(axis=1, keepdims=True)

        coords = coords - coords_min
        range_max = (
            (coords_max - coords_min).max(dim=-1, keepdim=True).values
            if isinstance(coords, torch.Tensor)
            else (coords_max - coords_min).max(axis=-1, keepdims=True)
        )
        coords = coords / (range_max + 1e-8)

        coords = (
            coords + (1 - coords.max(dim=1, keepdim=True).values) / 2
            if isinstance(coords, torch.Tensor)
            else coords + (1 - coords.max(axis=1, keepdims=True)) / 2
        )
        return coords
