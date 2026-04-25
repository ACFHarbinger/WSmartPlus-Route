"""cwcvrp.py module.

Attributes:
    CWCVRP: Capacitated WCVRP task, adding per-trip capacity enforcement to WCVRP.

Example:
    >>> import cwcvrp
"""

from .wcvrp import WCVRP


class CWCVRP(WCVRP):
    """Capacitated WCVRP.

    Attributes:
        NAME: Environment name identifier.
    """

    NAME = "cwcvrp"
