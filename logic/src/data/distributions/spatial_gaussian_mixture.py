"""spatial_gaussian_mixture.py module.

    Attributes:
        MODULE_VAR (Type): Description of module level variable.

    Example:
        >>> import spatial_gaussian_mixture
    """
from typing import Tuple

import torch


class Gaussian_Mixture:
    """Configurable Gaussian Mixture Model."""

    def __init__(self, num_modes: int = 0, cdist: int = 0):
        """Initialize Class.

        Args:
            num_modes (int): Description of num_modes.
            cdist (int): Description of cdist.
        """
        self.num_modes = num_modes
        self.cdist = cdist

    def sample(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """Sample.

        Args:
            size (Tuple[int, int, int]): Description of size.

        Returns:
            Any: Description of return value.
        """
        batch_size, num_loc, _ = size

        if self.num_modes == 0:
            return torch.rand((batch_size, num_loc, 2))
        elif self.num_modes == 1 and self.cdist == 1:
            return self._generate_gaussian(batch_size, num_loc)
        else:
            return torch.stack([self._generate_gaussian_mixture(num_loc) for _ in range(batch_size)])

    def _generate_gaussian_mixture(self, num_loc: int) -> torch.Tensor:
        """generate gaussian mixture.

        Args:
            num_loc (int): Description of num_loc.

        Returns:
            Any: Description of return value.
        """
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
        """generate gaussian.

        Args:
            batch_size (int): Description of batch_size.
            num_loc (int): Description of num_loc.

        Returns:
            Any: Description of return value.
        """
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
        """global min max scaling.

        Args:
            coords (torch.Tensor): Description of coords.

        Returns:
            Any: Description of return value.
        """
        coords_min = coords.min(0, keepdim=True).values
        coords_max = coords.max(0, keepdim=True).values
        coords = (coords - coords_min) / (coords_max - coords_min + 1e-8)

        return coords

    def _batch_normalize_and_center(self, coords: torch.Tensor) -> torch.Tensor:
        """batch normalize and center.

        Args:
            coords (torch.Tensor): Description of coords.

        Returns:
            Any: Description of return value.
        """
        coords_min = coords.min(dim=1, keepdim=True).values
        coords_max = coords.max(dim=1, keepdim=True).values

        coords = coords - coords_min
        range_max = (coords_max - coords_min).max(dim=-1, keepdim=True).values
        coords = coords / (range_max + 1e-8)

        coords = coords + (1 - coords.max(dim=1, keepdim=True).values) / 2

        return coords
