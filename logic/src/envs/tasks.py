"""
Legacy problem definitions for backward compatibility.
Provides the BaseProblem interface expected by legacy AttentionModel and Evaluator.
"""

from logic.src.constants.tasks import BIN_CAPACITY, COST_KM, REVENUE_KG, VEHICLE_CAPACITY
from logic.src.envs.tasks import (
    CVRPP,
    CWCVRP,
    SCWCVRP,
    SDWCVRP,
    VRPP,
    WCVRP,
    BaseProblem,
)

__all__ = [
    "BaseProblem",
    "COST_KM",
    "REVENUE_KG",
    "BIN_CAPACITY",
    "VEHICLE_CAPACITY",
    "VRPP",
    "CVRPP",
    "WCVRP",
    "CWCVRP",
    "SDWCVRP",
    "SCWCVRP",
]
