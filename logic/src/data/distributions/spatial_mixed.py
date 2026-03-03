"""spatial_mixed.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import spatial_mixed
"""

from typing import Optional, Tuple

import torch


class Mixed:
    """50% uniform + 50% Gaussian clusters."""

    def __init__(self, n_cluster_mix: int = 1):
        """Initialize Class.

        Args:
            n_cluster_mix (int): Description of n_cluster_mix.
        """
        self.n_cluster_mix = n_cluster_mix
        self.lower, self.upper = 0.2, 0.8
        self.std = 0.07

    def sample_tensor(self, size: Tuple[int, int, int], generator: Optional[torch.Generator] = None) -> torch.Tensor:
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

        center = self.lower + (self.upper - self.lower) * torch.rand(
            batch_size, self.n_cluster_mix * 2, generator=generator
        )

        coords = torch.FloatTensor(batch_size, num_loc, 2).uniform_(0, 1)
        mutate_idx = torch.stack(
            [torch.randperm(num_loc, generator=generator)[: num_loc // 2] for _ in range(batch_size)]
        )

        segment_size = num_loc // (2 * self.n_cluster_mix)
        remaining = num_loc // 2 - segment_size * (self.n_cluster_mix - 1)
        sizes = [segment_size] * (self.n_cluster_mix - 1) + [remaining]

        for i in range(self.n_cluster_mix):
            indices = mutate_idx[:, sum(sizes[:i]) : sum(sizes[: i + 1])]

            means_x = center[:, 2 * i].unsqueeze(1).expand(-1, sizes[i])
            means_y = center[:, 2 * i + 1].unsqueeze(1).expand(-1, sizes[i])

            new_points = torch.stack(
                [
                    torch.normal(means_x, self.std, generator=generator),
                    torch.normal(means_y, self.std, generator=generator),
                ],
                dim=2,
            )

            coords.scatter_(
                1,
                indices.unsqueeze(-1).expand(-1, -1, 2),
                new_points,
            )

        return coords.clamp_(0, 1)
