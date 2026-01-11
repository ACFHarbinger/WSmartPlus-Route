"""
Core problem definitions and dataset handlers for Vehicle Routing Problems.

This package provides the logic, state management, and dataset utilities for
various VRP flavors, including:
- VRP with Profits (VRPP, CVRPP)
- Waste Collection VRP (WCVRP, CWCVRP, SDWCVRP)
- Selective Capacitated Waste Collection VRP (SCWCVRP)
"""

from .vrpp.problem_vrpp import VRPP as VRPP, CVRPP as CVRPP
from .wcvrp.problem_wcvrp import WCVRP as WCVRP, CWCVRP as CWCVRP, SDWCVRP as SDWCVRP
from .swcvrp.problem_swcvrp import SCWCVRP as SCWCVRP
