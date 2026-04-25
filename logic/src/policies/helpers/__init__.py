"""
Policy Helpers Package.

This package provides specialized helper components for routing policies, including:
- `hpo`: Hyper-parameter optimization handlers and search spaces.
- `local_search`: Advanced local search managers (ACO, FILO, etc.).
- `operators`: Atomic inter-route and intra-route move implementations.
- `reinforcement_learning`: RL-specific policy helpers (Q-Learning, etc.).
- `solvers_and_matheuristics`: Specialized matheuristics and exact sub-solvers.

Attributes:
    hpo: Hyper-parameter optimization handlers.
    local_search: Local search managers.
    operators: Atomic inter-route and intra-route move implementations.
    reinforcement_learning: RL-specific policy helpers.
    solvers_and_matheuristics: Specialized matheuristics and exact sub-solvers.

Example:
    >>> import logic
    >>> solver = logic.get_policy_by_name("my_policy")
    >>> solver.evaluate()
    0.5678
"""

from . import hpo as hpo
from . import local_search as local_search
from . import operators as operators
from . import reinforcement_learning as reinforcement_learning
from . import solvers_and_matheuristics as solvers_and_matheuristics

__all__ = [
    "solvers_and_matheuristics",
    "hpo",
    "local_search",
    "operators",
    "reinforcement_learning",
]
