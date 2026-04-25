"""
Waste generation utilities for VRP instances.

This module provides functions for generating waste/fill values using various
statistical and spatial distributions. It delegates to distribution classes
from ``logic.src.data.distributions``.

Attributes:
    generate_waste: Generates waste values based on a distribution or empirical data.

Example:
    from logic.src.data.generators.waste import generate_waste
    waste = generate_waste(10, "gamma1", (coords, edges))
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
        problem_size: Description of problem_size.
        distribution: Description of distribution.
        graph: Description of graph.
        dataset_size: Description of dataset_size.
        grid: Description of grid.
        rng: Description of rng.
        sample_method: Description of sample_method.
        kwargs: Description of kwargs.

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

    # Automatically determine the best sampling method based on RNG type to avoid assertion errors
    if isinstance(rng, torch.Generator):
        sample_method = "sample_tensor"
    elif isinstance(rng, np.random.Generator):
        sample_method = "sample_array"

    return dist_obj.set_sampling_method(sample_method).sample(size, rng=rng)  # type: ignore[arg-type]
