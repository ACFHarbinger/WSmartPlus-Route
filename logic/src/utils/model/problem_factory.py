"""
Factory functions for loading problem classes.
"""

from __future__ import annotations

from typing import Any, Type


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
    from logic.src.envs.problems import (
        CVRPP,
        CWCVRP,
        SCWCVRP,
        SDWCVRP,
        VRPP,
        WCVRP,
    )

    problem = {
        "vrpp": VRPP,
        "cvrpp": CVRPP,
        "wcvrp": WCVRP,
        "cwcvrp": CWCVRP,
        "sdwcvrp": SDWCVRP,
        "scwcvrp": SCWCVRP,
    }.get(name)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem
