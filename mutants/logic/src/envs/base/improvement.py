"""
Improvement-based environment base class.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from tensordict import TensorDict

from .base import RL4COEnvBase


class ImprovementEnvBase(RL4COEnvBase):
    """
    Base environment for improvement-based methods.

    Improvement methods start with an initial solution and iteratively
    improve it through local search operations.
    """

    name: str = "improvement_base"

    @abstractmethod
    def _get_initial_solution(self, td: TensorDict) -> torch.Tensor:
        """
        Generate initial solution for improvement.
        """
        raise NotImplementedError

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """
        Reset with initial solution generation.
        """
        # Generate initial solution
        td["solution"] = self._get_initial_solution(td)

        return td
