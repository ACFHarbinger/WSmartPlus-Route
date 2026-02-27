"""
Statistical sampling distributions - Gamma.
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from logic.src.constants.data import GAMMA_PRESETS


class Gamma:
    """Gamma distribution sampling with optional per-node preset parameters."""

    def __init__(
        self,
        alpha: Union[float, torch.Tensor] = 2.0,
        theta: Union[float, torch.Tensor] = 2.0,
        option: Optional[int] = None,
    ):
        """Initialize Class.

        When ``option`` is provided, ``alpha`` and ``theta`` are ignored
        and per-node heterogeneous parameters are loaded from ``GAMMA_PRESETS``.

        Args:
            alpha (Union[float, torch.Tensor]): Shape parameter (ignored if ``option`` is set).
            theta (Union[float, torch.Tensor]): Scale parameter (ignored if ``option`` is set).
            option (Optional[int]): Preset index (0-3) for per-node heterogeneous params.
        """
        self.option = option
        self.alpha = alpha
        self.theta = theta

    def _tile_param(self, size: int, param: List[int]) -> List[int]:
        """
        Tile a parameter list to match a target size.

        Args:
            size: Target size.
            param: Parameter list to tile.

        Returns:
            Tiled parameter list of length ``size``.
        """
        param_len = len(param)
        if size == param_len:
            return param

        param = param * math.ceil(size / param_len)
        if size % param_len != 0:
            param = param[: param_len - size % param_len]
        return param

    def sample_tensor(self, size: Tuple[int, ...]) -> torch.Tensor:
        """Sample from Gamma distribution.

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc))

        Returns:
            torch.Tensor: Sampled values
        """
        if self.option is not None:
            # Use per-node heterogeneous params; fall back to numpy then convert
            return torch.from_numpy(self.sample_array(size)).float()

        m = torch.distributions.Gamma(self.alpha, 1 / self.theta)
        return m.sample(torch.Size(size))

    def sample_array(self, size: Tuple[int, ...]) -> np.ndarray:
        """Sample from Gamma distribution.

        When ``option`` is set, uses per-node heterogeneous alpha/theta
        parameters tiled to match ``problem_size`` (the last dimension of ``size``).

        Args:
            size: Sampling shape (e.g., (batch_size, num_loc))

        Returns:
            np.ndarray: Sampled values
        """
        if self.option is not None:
            alpha_pattern, theta_pattern = GAMMA_PRESETS[self.option]
            problem_size = size[-1]
            k = self._tile_param(problem_size, alpha_pattern)
            th = self._tile_param(problem_size, theta_pattern)
            return np.random.gamma(k, th, size=size) / 100.0

        return np.random.gamma(self.alpha, self.theta, size=size) / 100.0
