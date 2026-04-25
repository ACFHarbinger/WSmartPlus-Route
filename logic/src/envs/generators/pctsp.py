"""
PCTSP / SPCTSP problem generator.

Following Kool et al. (2019): penalty and prize structures designed so that
roughly half of the nodes are visited on average.

Attributes:
    PCTSPGenerator: PCTSPGenerator class.

Example:
    >>> from logic.src.envs.generators import PCTSPGenerator
    >>> generator = PCTSPGenerator(num_loc=20)
    >>> instance = generator.generate()
    >>> instance
    TensorDict({
        'locs': TensorDict(
            # ... customer locations (shape: [*B, 20, 2]) ...
        )
        # ... other fields ...
    })
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch
from tensordict import TensorDict

from .base import Generator

# Penalty scale table reuses the max-length table (same values as OP)
MAX_PENALTIES: dict[int, float] = {20: 2.0, 50: 3.0, 100: 4.0}


class PCTSPGenerator(Generator):
    """
    Generator for the Prize-Collecting TSP (PCTSP / SPCTSP).

    Produces instances with:
    - locs               : [*B, num_loc, 2]  — customer coordinates
    - depot              : [*B, 2]           — depot coordinate
    - penalty            : [*B, num_loc]     — per-node skip penalty
    - deterministic_prize: [*B, num_loc]     — expected prize (always known)
    - stochastic_prize   : [*B, num_loc]     — actual prize revealed on visit
                                               (SPCTSP only; same as det. in PCTSP)

    Attributes:
        penalty_factor: Scale factor applied to the maximum penalty.
        prize_required: Minimum total prize the agent must collect before returning.
        max_penalty: Computed maximum per-node skip penalty.
        depot_type: Depot placement strategy ("random", "center", or "corner").
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[str, Callable] = "uniform",
        penalty_factor: float = 3.0,
        prize_required: float = 1.0,
        depot_type: str = "random",
        device: Union[str, torch.device] = "cpu",
        rng: Optional[Any] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise PCTSPGenerator.

        Args:
            num_loc: Number of customer locations (excluding depot).
            min_loc: Lower bound for node coordinates.
            max_loc: Upper bound for node coordinates.
            loc_distribution: Spatial distribution.
            penalty_factor: Scales the maximum penalty (see Kool et al. 2019).
            prize_required: Minimum total prize the agent must collect before
                returning to the depot (normalised; default 1.0).
            depot_type: Depot placement ("center", "corner", "random").
            device: Target device.
            rng: Optional numpy RNG.
            generator: Optional torch RNG.
            kwargs: Forwarded to Generator base.
        """
        super().__init__(
            num_loc=num_loc,
            min_loc=min_loc,
            max_loc=max_loc,
            loc_distribution=loc_distribution,
            device=device,
            rng=rng,
            generator=generator,
            **kwargs,
        )
        self.penalty_factor = penalty_factor
        self.prize_required = prize_required
        self.depot_type = depot_type

        # Max penalty scales with num_loc so total penalty ≈ total tour length
        closest = min(MAX_PENALTIES.keys(), key=lambda x: abs(x - num_loc))
        base_max_pen = MAX_PENALTIES.get(num_loc, MAX_PENALTIES[closest])
        self.max_penalty = base_max_pen * penalty_factor / num_loc

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate a batch of PCTSP instances.

        Args:
            batch_size: Batch size.

        Returns:
            TensorDict with:
                locs: [*B, num_loc, 2]     — customer coordinates
                depot: [*B, 2]              — depot coordinate
                penalty: [*B, num_loc]        — penalty for not visiting each customer
                deterministic_prize: [*B, num_loc] — prize for visiting each customer
                stochastic_prize: [*B, num_loc] — stochastic prize
        """
        locs = self._generate_locations(batch_size)
        depot = self._generate_depot(batch_size)

        # Penalty: uniform in [0, max_penalty]
        penalty = torch.rand(*batch_size, self.num_loc, device=self.device, generator=self.generator) * self.max_penalty

        # Deterministic prize: uniform in [0, 4/num_loc]
        # Expectation = 2/num_loc, so sum ≈ 2 → agent visits ~half the nodes
        max_det_prize = 4.0 / self.num_loc
        deterministic_prize = (
            torch.rand(*batch_size, self.num_loc, device=self.device, generator=self.generator) * max_det_prize
        )

        # Stochastic prize: uniform in [0, 2*det_prize], mean = det_prize
        stochastic_prize = (
            torch.rand(*batch_size, self.num_loc, device=self.device, generator=self.generator)
            * 2.0
            * deterministic_prize
        )

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "penalty": penalty,
                "deterministic_prize": deterministic_prize,
                "stochastic_prize": stochastic_prize,
            },
            batch_size=batch_size,
            device=self.device,
        )

    def _generate_depot(self, batch_size: tuple[int, ...]) -> torch.Tensor:
        """Generate depot location.

        Args:
            batch_size: Batch size.

        Returns:
            Depot location.
        """
        if self.depot_type == "center":
            center = (self.max_loc + self.min_loc) / 2.0
            return torch.full((*batch_size, 2), center, device=self.device, dtype=torch.float32)
        elif self.depot_type == "corner":
            return torch.full((*batch_size, 2), self.min_loc, device=self.device, dtype=torch.float32)
        else:  # random
            return self._uniform_locations(batch_size, 1).squeeze(-2)
