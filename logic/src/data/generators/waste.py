"""
Waste generation utilities for VRP instances.

This module provides functions for generating waste/fill values using various
statistical and spatial distributions. It delegates to distribution classes
from ``logic.src.data.distributions``.
"""

from typing import Any, Optional, Tuple, Union

import numpy as np

from logic.src.data.distributions import Beta, Constant, Distance, Gamma, Uniform


def generate_waste(
    problem_size: int,
    distribution: str,
    graph: Tuple[Any, Any],
    dataset_size: int = 1,
    grid: Optional[Any] = None,
    rng: Optional[np.random.RandomState] = None,
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
        **kwargs: Additional parameters (e.g., alpha, beta for beta distribution).

    Returns:
        Generated waste values.
    """
    size = (dataset_size, problem_size)
    if distribution == "empty":
        wp: Any = Constant(value=0.0).sample_array(size, rng=rng)
    elif distribution == "const":
        wp = Constant(value=1.0).sample_array(size, rng=rng)
    elif distribution in ["unif", "uniform"]:
        wp = Uniform().sample_array(size, rng=rng)
    elif "gamma" in distribution:
        try:
            digit = "".join([c for c in distribution if c.isdigit()])
            gamma_option = int(digit) - 1 if digit else 0
        except ValueError:
            gamma_option = 0
        wp = Gamma(option=gamma_option).sample_array(size, rng=rng)
    elif "beta" in distribution:
        alpha = kwargs.get("alpha", 0.5)
        beta_p = kwargs.get("beta", 0.5)
        wp = Beta(alpha=alpha, beta=beta_p).sample_array(size, rng=rng)
    elif "emp" in distribution:
        if grid is None:
            raise ValueError("grid must be provided for empirical distribution")
        sampled_value = grid.sample(n_samples=dataset_size, rng=rng)
        todaysfilling = np.maximum(sampled_value, 0)
        wp = np.minimum(todaysfilling, 100.0)
    else:
        assert distribution == "dist"
        wp = Distance(graph).sample_array()
        if dataset_size == 1:
            return wp[0] if wp.ndim > 1 else wp
        return wp

    if dataset_size == 1:
        return wp[0]
    return wp
