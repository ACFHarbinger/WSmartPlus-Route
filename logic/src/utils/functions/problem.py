"""
Problem-specific utility functions.
"""

from typing import Any


def is_wc_problem(problem: Any) -> bool:
    """
    Check if the problem is a Waste Collection (WC) variant.

    Args:
        problem: Problem instance or name string.

    Returns:
        bool: True if it's a WC variant.
    """
    name = problem if isinstance(problem, str) else getattr(problem, "NAME", "")
    name = name.lower()
    return any(wc_tag in name for wc_tag in ["wcvrp", "swcvrp", "sdwcvrp"])


def is_vrpp_problem(problem: Any) -> bool:
    """
    Check if the problem is a Vehicle Routing Problem with Profits (VRPP) variant.

    Args:
        problem: Problem instance or name string.

    Returns:
        bool: True if it's a VRPP variant.
    """
    name = problem if isinstance(problem, str) else getattr(problem, "NAME", "")
    name = name.lower()
    return any(vrpp_tag in name for vrpp_tag in ["vrpp", "cvrpp", "pcvrp"])


def is_tsp_problem(problem: Any) -> bool:
    """
    Check if the problem is a Traveling Salesperson Problem (TSP) variant.

    Args:
        problem: Problem instance or name string.

    Returns:
        bool: True if it's a TSP variant.
    """
    name = problem if isinstance(problem, str) else getattr(problem, "NAME", "")
    name = name.lower()
    return any(tsp_tag in name for tsp_tag in ["tsp", "atsp"])
