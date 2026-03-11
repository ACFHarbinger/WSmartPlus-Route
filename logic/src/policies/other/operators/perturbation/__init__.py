"""
Perturbation Operators Package.

This package contains operator implementations for perturbing the current
solution to escape local optima.

Attributes:
    kick (function): Large-scale neighborhood change (destroy & repair).
    perturb (function): Small-scale random changes.

Example:
    >>> from logic.src.policies.other.operators.perturbation import kick
    >>> kick(context, destroy_ratio=0.2)
"""

from .kick import kick
from .perturb import perturb
from .unstringing_stringing_i import apply_type_i_us
from .unstringing_stringing_ii import apply_type_ii_us
from .unstringing_stringing_iii import apply_type_iii_us
from .unstringing_stringing_iv import apply_type_iv_us

__all__ = [
    "perturb",
    "kick",
    "apply_type_i_us",
    "apply_type_ii_us",
    "apply_type_iii_us",
    "apply_type_iv_us",
]
