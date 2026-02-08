"""tag.py module.

    Attributes:
        MODULE_VAR (Type): Description of module level variable.

    Example:
        >>> import tag
    """
from enum import Enum


class TAG(Enum):
    """Quality tags indicating container data reliability level."""

    LOW_MEASURES = 0
    INSIDE_BOX = 1
    OK = 2
    WARN = 3
    LOCAL_WARN = 4
