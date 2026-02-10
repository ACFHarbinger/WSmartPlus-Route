"""cwcvrp.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import cwcvrp
"""

from .wcvrp import WCVRP


class CWCVRP(WCVRP):
    """Capacitated WCVRP."""

    NAME = "cwcvrp"
