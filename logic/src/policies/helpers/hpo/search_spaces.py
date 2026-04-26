"""
Search space definitions for simulation policies.

This module contains the default hyperparameter search spaces for all supported
simulation policies, including ALNS, HGS, ACO, and SISR. Each policy's search
space is defined as a dictionary with parameter names as keys and their HPO
specifications as values.

Attributes:
    POLICY_SEARCH_SPACES: Dictionary mapping policy names to their search spaces.

Functions:
    get_search_space: Retrieves the search space for a specific policy.

Example:
    >>> from logic.src.policies.helpers.hpo import get_search_space
    >>> alns_space = get_search_space("alns")
    >>> print(alns_space)
    {
        'max_iterations': {'type': 'int', 'low': 500, 'high': 10000, 'step': 500},
        'start_temp': {'type': 'float', 'low': 1.0, 'high': 500.0, 'log': True},
        ...
    }
"""

from typing import Any, Dict

# Default search spaces for common simulation policies
POLICY_SEARCH_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "alns": {
        "max_iterations": {"type": "int", "low": 500, "high": 10000, "step": 500},
        "start_temp": {"type": "float", "low": 1.0, "high": 500.0, "log": True},
        "cooling_rate": {"type": "float", "low": 0.9, "high": 0.999},
        "reaction_factor": {"type": "float", "low": 0.01, "high": 0.5},
        "max_removal_pct": {"type": "float", "low": 0.1, "high": 0.5},
    },
    "hgs": {
        "mu": {"type": "int", "low": 10, "high": 200, "step": 10},
        "nb_elite": {"type": "int", "low": 2, "high": 50, "step": 2},
        "mutation_rate": {"type": "float", "low": 0.01, "high": 0.4},
        "crossover_rate": {"type": "float", "low": 0.4, "high": 0.9},
        "nb_granular": {"type": "int", "low": 5, "high": 50},
        "local_search_iterations": {"type": "int", "low": 100, "high": 2000, "step": 100},
    },
    "aco": {
        "n_ants": {"type": "int", "low": 5, "high": 100, "step": 5},
        "alpha": {"type": "float", "low": 0.1, "high": 5.0},
        "beta": {"type": "float", "low": 0.1, "high": 10.0},
        "rho": {"type": "float", "low": 0.01, "high": 0.5},
        "scale": {"type": "float", "low": 1.0, "high": 10.0},
    },
    "sisr": {
        "max_iterations": {"type": "int", "low": 100, "high": 5000, "step": 100},
        "start_temp": {"type": "float", "low": 1.0, "high": 500.0, "log": True},
        "cooling_rate": {"type": "float", "low": 0.9, "high": 0.999},
        "max_string_len": {"type": "int", "low": 2, "high": 20},
        "avg_string_len": {"type": "float", "low": 1.0, "high": 10.0},
        "destroy_ratio": {"type": "float", "low": 0.05, "high": 0.4},
    },
}


def get_search_space(policy_name: str) -> Dict[str, Dict[str, Any]]:
    """Get the search space for a specific policy.

    Args:
        policy_name (str): Name of the policy (e.g., 'alns', 'hgs').

    Returns:
        Dict[str, Dict[str, Any]]: The search space specification dictionary for
            the requested policy, or an empty dictionary if not found.
    """
    return POLICY_SEARCH_SPACES.get(policy_name.lower(), {})
