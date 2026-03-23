"""
Factory functions for loading problem classes.
"""

from __future__ import annotations

from typing import Any, Type

from logic.src.envs.problems import (
    CVRPP,
    CWCVRP,
    SCWCVRP,
    VRPP,
    WCVRP,
)


def load_problem(name: str) -> Type[Any]:
    """
    Factory function to load a problem class by name.

    Args:
        name: The problem name (e.g., 'vrpp', 'wcvrp').

    Returns:
        The problem class.

    Raises:
        AssertionError: If problem name is unsupported.
    """
    problem = {
        "vrpp": VRPP,
        "cvrpp": CVRPP,
        "wcvrp": WCVRP,
        "cwcvrp": CWCVRP,
        "scwcvrp": SCWCVRP,
    }.get(name)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem
