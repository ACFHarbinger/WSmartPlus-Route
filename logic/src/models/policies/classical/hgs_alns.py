"""
HGS-ALNS Hybrid Policy for RL4CO.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from logic.src.models.policies.classical.hgs import VectorizedHGS
from logic.src.policies.hgs_alns_solver import HGSALNSSolver
from logic.src.policies.hgs_aux.types import HGSParams


class VectorizedHGSALNS(VectorizedHGS, BaseModel):
    """
    HGS-based Policy wrapper that uses ALNS for education phase.

    This implementation combines Pydantic for parameter management with
    the logic of HGSPolicy for RL4CO integration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Policy parameters as Pydantic fields
    env_name: Optional[str] = Field("vrpp", description="Name of the environment")
    time_limit: float = Field(5.0, description="HGS search time limit in seconds")
    population_size: int = Field(50, description="Target population size")
    n_generations: int = Field(50, description="Number of HGS generations")
    elite_size: int = Field(5, description="Number of elite individuals")
    max_vehicles: int = Field(0, description="Maximum number of vehicles")
    alns_education_iterations: int = Field(50, description="Number of ALNS iterations for education phase")

    def __init__(self, **data: Any):
        """Initialize HGSALNSPolicy."""
        # Initialize BaseModel first to populate fields
        BaseModel.__init__(self, **data)

        # Initialize HGSPolicy (nn.Module) using data from fields
        VectorizedHGS.__init__(
            self,
            env_name=self.env_name,
            time_limit=self.time_limit,
            population_size=self.population_size,
            n_generations=self.n_generations,
            elite_size=self.elite_size,
            max_vehicles=self.max_vehicles,
            **data,
        )

    def solve(self, dist_matrix, demands, capacity, **kwargs):
        """
        Solve a single instance using the scalar HGSALNSSolver.

        This overrides the standard solve logic to use the hybrid solver.
        """
        # Convert torch tensors to numpy if needed
        import torch

        if isinstance(dist_matrix, torch.Tensor):
            dist_matrix = dist_matrix.cpu().numpy()
        if isinstance(demands, torch.Tensor):
            # If batch dim is present, take first instance
            if demands.dim() > 1:
                demands = demands[0]
            demands_dict = {i: demands[i].item() for i in range(len(demands))}
        else:
            demands_dict = demands

        params = HGSParams(
            time_limit=int(self.time_limit),
            population_size=self.population_size,
            elite_size=self.elite_size,
            max_vehicles=self.max_vehicles,
        )

        solver = HGSALNSSolver(
            dist_matrix=dist_matrix,
            demands=demands_dict,
            capacity=float(capacity),
            R=kwargs.get("R", 1.0),
            C=kwargs.get("C", 1.0),
            params=params,
            alns_education_iterations=self.alns_education_iterations,
        )

        return solver.solve()
