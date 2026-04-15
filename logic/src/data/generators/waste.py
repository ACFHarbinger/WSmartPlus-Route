"""
Waste generation utilities for VRP instances.

This module provides functions for generating waste/fill values using various
statistical and spatial distributions. It delegates to distribution classes
from ``logic.src.data.distributions``.
"""

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch

from logic.src.data.distributions import (
    BaseDistribution,
    Beta,
    Constant,
    Distance,
    Empirical,
    Gamma,
    Uniform,
)


def generate_waste(
    problem_size: int,
    distribution: str,
    graph: Tuple[Any, Any],
    dataset_size: int = 1,
    grid: Optional[Any] = None,
    rng: Optional[Union[torch.Generator, np.random.Generator]] = None,
    sample_method: Optional[str] = "sample_array",
    **kwargs: Any,
) -> Union[np.ndarray, Any]:
    """
    Generates waste values based on a distribution or empirical data.

    Delegates to distribution classes from ``logic.src.data.distributions``
    for standard distributions.

    Args:
        problem_size: Number of nodes/bins.
        distribution: Distribution type ('empty', 'const', 'unif', 'gammaX', 'emp', 'dist', 'beta').
        graph: (depot, loc) coordinates.
        dataset_size: Number of datasets to generate. Defaults to 1.
        grid: GridBase object for empirical sampling.
        rng: Random number generator.
        **kwargs: Additional parameters (e.g., alpha, beta for beta distribution).

    Returns:
        Generated waste values.
    """
    size = (dataset_size, problem_size)
    dist_obj: Optional[BaseDistribution] = None
    if distribution == "empty":
        dist_obj = Constant(value=0.0)
    elif distribution == "const":
        dist_obj = Constant(value=1.0)
    elif distribution in ["unif", "uniform"]:
        low = kwargs.get("low", 0)
        high = kwargs.get("high", 1)
        dist_obj = Uniform(low=low, high=high)
    elif "gamma" in distribution:
        try:
            digit = "".join([c for c in distribution if c.isdigit()])
            gamma_option = int(digit) - 1 if digit else 0
        except ValueError:
            gamma_option = 0
        dist_obj = Gamma(option=gamma_option)
    elif "beta" in distribution:
        alpha = kwargs.get("alpha", 0.5)
        beta_p = kwargs.get("beta", 0.5)
        dist_obj = Beta(alpha=alpha, beta=beta_p)
    elif "emp" in distribution:
        if grid is None:
            raise ValueError("grid must be provided for empirical distribution")
        dist_obj = Empirical(grid=grid)
    else:
        assert distribution == "dist"
        dist_obj = Distance(graph)

    return dist_obj.set_sampling_method(sample_method).sample(size, rng=rng)  # type: ignore[arg-type]
