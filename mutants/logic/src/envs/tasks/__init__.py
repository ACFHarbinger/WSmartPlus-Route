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
from .cwcvrp import CWCVRP
from .scwcvrp import SCWCVRP
from .sdwcvrp import SDWCVRP
from .vrp import CVRPP, VRPP
from .wcvrp import WCVRP

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
