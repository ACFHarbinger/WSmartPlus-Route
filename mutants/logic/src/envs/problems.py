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
from logic.src.envs.tasks.cvrpp import CVRPP
from logic.src.envs.tasks.cwcvrp import CWCVRP
from logic.src.envs.tasks.scwcvrp import SCWCVRP
from logic.src.envs.tasks.sdwcvrp import SDWCVRP
from logic.src.envs.tasks.vrpp import VRPP
from logic.src.envs.tasks.wcvrp import WCVRP

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
