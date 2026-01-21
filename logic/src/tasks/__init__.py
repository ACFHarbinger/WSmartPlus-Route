"""
Core problem definitions and dataset handlers for Vehicle Routing Problems.

This package provides the logic, state management, and dataset utilities for
various VRP flavors, including:
- VRP with Profits (VRPP, CVRPP)
- Waste Collection VRP (WCVRP, CWCVRP, SDWCVRP)
- Selective Capacitated Waste Collection VRP (SCWCVRP)
"""

from .swcvrp.problem_swcvrp import SCWCVRP as SCWCVRP
from .vrpp.problem_vrpp import CVRPP as CVRPP
from .vrpp.problem_vrpp import VRPP as VRPP
from .wcvrp.problem_wcvrp import CWCVRP as CWCVRP
from .wcvrp.problem_wcvrp import SDWCVRP as SDWCVRP
from .wcvrp.problem_wcvrp import WCVRP as WCVRP
