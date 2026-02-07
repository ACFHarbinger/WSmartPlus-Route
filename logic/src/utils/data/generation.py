from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple, Union

import numpy as np


def _get_fill_gamma(dataset_size: int, problem_size: int, gamma_option: int) -> np.ndarray:
    """
    Generates waste levels using a Gamma distribution.

    Args:
        dataset_size: Number of instances.
        problem_size: Number of bins per instance.
        gamma_option: Index to select gamma parameters (alpha, theta).

    Returns:
        Generated waste levels [dataset_size, problem_size].
    """

    def __set_distribution_param(size: int, param: List[int]) -> List[int]:
        """
        Sets a distribution parameter if not already present.

        Args:
            size: Size to match.
            param: Parameter list.

        Returns:
            Adjusted parameter list.
        """
        param_len = len(param)
        if size == param_len:
            return param

        param = param * math.ceil(size / param_len)
        if size % param_len != 0:
            param = param[: param_len - size % param_len]
        return param

    if gamma_option == 0:
        alpha = [5, 5, 5, 5, 5, 10, 10, 10, 10, 10]
        theta = [5, 2]
    elif gamma_option == 1:
        alpha = [2, 2, 2, 2, 2, 6, 6, 6, 6, 6]
        theta = [6, 4]
    elif gamma_option == 2:
        alpha = [1, 1, 1, 1, 1, 3, 3, 3, 3, 3]
        theta = [8, 6]
    else:
        assert gamma_option == 3
        alpha = [5, 2]
        theta = [10]

    k = __set_distribution_param(problem_size, alpha)
    th = __set_distribution_param(problem_size, theta)
    return np.random.gamma(k, th, size=(dataset_size, problem_size)) / 100.0


def generate_waste_prize(
    problem_size: int,
    distribution: str,
    graph: Tuple[Any, Any],
    dataset_size: int = 1,
    bins: Optional[Any] = None,
) -> Union[np.ndarray, Any]:  # typed Any to avoid torch dependency here if possible, but callers expect tensor/array
    """
    Generates waste or prize values based on a distribution or empirical data.

    Args:
        problem_size: Number of nodes/bins.
        distribution: Distribution type ('empty', 'const', 'unif', 'gammaX', 'emp', 'dist').
        graph: (depot, loc) coordinates.
        dataset_size: Number of datasets to generate. Defaults to 1.
        bins: Bins object for empirical sampling.

    Returns:
        Generated waste/prizes.
    """
    if distribution == "empty":
        wp: Any = np.zeros(shape=(dataset_size, problem_size))
    elif distribution == "const":
        wp = np.ones(shape=(dataset_size, problem_size))
    elif distribution in ["unif", "uniform"]:
        wp = (1 + np.random.randint(0, 100, size=(dataset_size, problem_size))) / 100.0
    elif "gamma" in distribution:
        gamma_option = int(distribution[-1]) - 1
        wp = _get_fill_gamma(dataset_size, problem_size, gamma_option)
    elif "emp" in distribution:
        if bins is None:
            raise ValueError("bins must be provided for empirical distribution")
        wp = bins.stochasticFilling(n_samples=dataset_size, only_fill=True) / 100.0
    else:
        assert distribution == "dist"
        depot, loc = graph
        if dataset_size > 1:
            wp = np.linalg.norm(depot[:, None, :] - loc, axis=-1)
            return (1 + (wp / wp.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.0
        else:
            # Consistent numpy handling matching the >1 case
            wp = np.linalg.norm(depot[:, None, :] - loc, axis=-1)
            return (1 + (wp / wp.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.0

    if dataset_size == 1:
        return wp[0]
    return wp
