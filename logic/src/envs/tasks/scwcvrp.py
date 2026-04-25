"""scwcvrp.py module.

Attributes:
    SCWCVRP: Stochastic Capacitated WCVRP task, extending WCVRP with uncertainty modeling.

Example:
    >>> import scwcvrp
"""

from .wcvrp import WCVRP


class SCWCVRP(WCVRP):
    """Selective Capacitated WCVRP.

    Attributes:
        NAME: Environment name identifier.
    """

    NAME = "scwcvrp"
