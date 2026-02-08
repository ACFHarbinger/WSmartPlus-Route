"""scwcvrp.py module.

    Attributes:
        MODULE_VAR (Type): Description of module level variable.

    Example:
        >>> import scwcvrp
    """
from .wcvrp import WCVRP


class SCWCVRP(WCVRP):
    """Selective Capacitated WCVRP."""

    NAME = "scwcvrp"
