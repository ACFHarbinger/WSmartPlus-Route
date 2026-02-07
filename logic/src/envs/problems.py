"""
Facade for the problems package.
"""

from logic.src.constants.tasks import (
    BIN_CAPACITY,
    COST_KM,
    REVENUE_KG,
    VEHICLE_CAPACITY,
)
from logic.src.envs.tasks.base import BaseProblem
from logic.src.envs.tasks.vrp import CVRPP, VRPP
from logic.src.envs.tasks.waste import CWCVRP, SCWCVRP, SDWCVRP, WCVRP

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
