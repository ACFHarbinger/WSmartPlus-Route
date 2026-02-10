"""__init__.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import __init__
"""

from .datasets import generate_datasets
from .validators import validate_gen_data_args

__all__ = ["generate_datasets", "validate_gen_data_args"]
