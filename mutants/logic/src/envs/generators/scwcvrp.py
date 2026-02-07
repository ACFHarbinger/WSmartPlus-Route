"""
SCWCVRP problem generator.
"""

from __future__ import annotations

from typing import Any, Callable, Union

import torch
from tensordict import TensorDict

from .wcvrp import WCVRPGenerator


class SCWCVRPGenerator(WCVRPGenerator):
    """
    Generator for Stochastic Capacitated Waste Collection VRP (SCWCVRP) instances.

    Adds noise to the fill levels to simulate uncertain demand.
    """

    def __init__(
        self,
        num_loc: int = 50,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[str, Callable] = "uniform",
        min_fill: float = 0.0,
        max_fill: float = 1.0,
        fill_distribution: str = "uniform",
        capacity: float = 100.0,
        cost_km: float = 1.0,
        revenue_kg: float = 0.1625,
        depot_type: str = "center",
        noise_mean: float = 0.0,
        noise_variance: float = 0.0,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        """Initialize SCWCVRP generator."""
        super().__init__(
            num_loc=num_loc,
            min_loc=min_loc,
            max_loc=max_loc,
            loc_distribution=loc_distribution,
            min_fill=min_fill,
            max_fill=max_fill,
            fill_distribution=fill_distribution,
            capacity=capacity,
            cost_km=cost_km,
            revenue_kg=revenue_kg,
            depot_type=depot_type,
            device=device,
            **kwargs,
        )
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate SCWCVRP instances."""
        td = super()._generate(batch_size)

        # Rename 'demand' to 'real_waste' (internal) and add 'noisy_waste'
        real_waste = td["demand"].clone()
        td["real_waste"] = real_waste

        if self.noise_variance > 0:
            noise = torch.normal(
                mean=self.noise_mean,
                std=self.noise_variance**0.5,
                size=real_waste.size(),
                device=self.device,
            )
            noisy_waste = (real_waste + noise).clamp(min=0.0, max=1.0)
        else:
            noisy_waste = real_waste.clone()

        td["waste"] = noisy_waste
        td["demand"] = noisy_waste  # Set demand to the noisy version for the agent

        return td
