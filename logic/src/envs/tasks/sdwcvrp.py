"""sdwcvrp.py module.

    Attributes:
        MODULE_VAR (Type): Description of module level variable.

    Example:
        >>> import sdwcvrp
    """
from .wcvrp import WCVRP


class SDWCVRP(WCVRP):
    """Stochastic Demand WCVRP."""

    NAME = "sdwcvrp"
