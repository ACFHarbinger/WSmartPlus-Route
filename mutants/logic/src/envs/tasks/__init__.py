"""
Problem definitions sub-package.
"""

from logic.src.constants.tasks import (
    BIN_CAPACITY,
    COST_KM,
    REVENUE_KG,
    VEHICLE_CAPACITY,
)

from .base import BaseProblem
from .vrp import CVRPP, VRPP
from .waste import CWCVRP, SCWCVRP, SDWCVRP, WCVRP

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
