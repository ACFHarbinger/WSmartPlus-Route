"""
ATSP problem generator.

Generates asymmetric distance matrices satisfying the triangle inequality
(TMAT class) following the approach of Kwon et al. (MatNet, 2021).
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch
from tensordict import TensorDict

from .base import Generator


class ATSPGenerator(Generator):
    """
    Generator for the Asymmetric Traveling Salesman Problem (ATSP).

    Produces random asymmetric distance matrices of shape
    [*B, num_loc, num_loc] with zero diagonal.  When ``tmat_class=True``
    (default) the triangle inequality is enforced via iterative relaxation.

    Instance layout
    ---------------
    - cost_matrix : [*B, num_loc, num_loc] — asymmetric pairwise distances
    """

    def __init__(
        self,
        num_loc: int = 10,
        min_dist: float = 0.0,
        max_dist: float = 1.0,
        tmat_class: bool = True,
        device: Union[str, torch.device] = "cpu",
        rng: Optional[Any] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise ATSPGenerator.

        Args:
            num_loc: Number of nodes (no depot in ATSP).
            min_dist: Lower bound for distance values.
            max_dist: Upper bound for distance values.
            tmat_class: Enforce triangle inequality (True = TMAT class).
            device: Target device for generated tensors.
            rng: Optional numpy RNG (forwarded to base).
            generator: Optional torch RNG.
            **kwargs: Forwarded to Generator base.
        """
        super().__init__(num_loc=num_loc, device=device, rng=rng, generator=generator, **kwargs)
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.tmat_class = tmat_class

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate a batch of ATSP distance matrices."""
        dms = (
            torch.rand(*batch_size, self.num_loc, self.num_loc, device=self.device, generator=self.generator)
            * (self.max_dist - self.min_dist)
            + self.min_dist
        )
        # Zero diagonal (self-travel cost)
        idx = torch.arange(self.num_loc, device=self.device)
        dms[..., idx, idx] = 0.0
        # Enforce triangle inequality
        if self.tmat_class:
            for i in range(self.num_loc):
                dms = torch.minimum(dms, dms[..., :, [i]] + dms[..., [i], :])
        return TensorDict({"cost_matrix": dms}, batch_size=batch_size, device=self.device)
