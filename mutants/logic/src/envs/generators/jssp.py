"""
Job Shop Scheduling Problem (JSSP) instance generator.
"""

from __future__ import annotations

from typing import Any, Union

import torch
from tensordict import TensorDict

from .base import Generator


class JSSPGenerator(Generator):
    """
    Generator for Job Shop Scheduling Problem (JSSP) instances.

    Generates random JSSP instances where each job consists of M operations
    to be processed on M different machines in a specific order.

    Features:
    - proc_time: Processing time for each operation (Job j, Op i)
    - machine_order: Machine required for each operation (Job j, Op i) -> Machine ID
    """

    def __init__(
        self,
        num_jobs: int = 10,
        num_machines: int = 10,
        min_duration: int = 1,
        max_duration: int = 99,
        duration_distribution: str = "uniform",
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        """
        Initialize JSSP generator.

        Args:
            num_jobs: Number of jobs (J).
            num_machines: Number of machines (M).
            min_duration: Minimum processing time.
            max_duration: Maximum processing time.
            duration_distribution: Distribution for durations ("uniform").
            device: Device to place tensors on.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            num_loc=num_jobs,  # Reusing num_loc to store num_jobs if needed, or ignored
            min_loc=min_duration,
            max_loc=max_duration,
            loc_distribution=duration_distribution,
            device=device,
            **kwargs,
        )
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.min_duration = min_duration
        self.max_duration = max_duration

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate JSSP instances."""
        # Random durations
        proc_time = torch.randint(
            self.min_duration,
            self.max_duration + 1,
            (*batch_size, self.num_jobs, self.num_machines),
            device=self.device,
            dtype=torch.float32,
        )

        # Generate machine order: (batch, J, M)
        # For each job, the sequence of machines is a random permutation of 0..M-1
        # We need to generate a permutation for each job in the batch

        # 1. Create base indices [0, ..., M-1]
        # 2. Shuffle them for each job

        # shape: (batch * J, M)
        raw_perms = torch.rand((*batch_size, self.num_jobs, self.num_machines), device=self.device).argsort(dim=-1)
        machine_order = raw_perms

        return TensorDict(
            {
                "proc_time": proc_time,
                "machine_order": machine_order,
                "num_jobs": torch.full((*batch_size,), self.num_jobs, device=self.device),
                "num_machines": torch.full((*batch_size,), self.num_machines, device=self.device),
            },
            batch_size=batch_size,
            device=self.device,
        )
