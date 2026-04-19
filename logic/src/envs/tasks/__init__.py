"""
Problem definitions sub-package.
"""

from logic.src.constants.tasks import (
    BIN_CAPACITY,
    COST_KM,
    REVENUE_KG,
    VEHICLE_CAPACITY,
)

from .atsp import ATSP
from .base import BaseProblem
from .cvrp import CVRP
from .cvrpp import CVRPP
from .cwcvrp import CWCVRP
from .irp import IRP
from .op import OP
from .pctsp import PCTSP
from .pdp import PDP
from .scwcvrp import SCWCVRP
from .spctsp import SPCTSP
from .thop import ThOP
from .tsp import TSP
from .vrpp import VRPP
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
    "SCWCVRP",
    "IRP",
    "ATSP",
    "TSP",
    "CVRP",
    "OP",
    "PCTSP",
    "SPCTSP",
    "PDP",
    "ThOP",
]
