"""
Data generation tools for creating synthetic VRP instances.

Provides builders, dataset generation scripts, and validators for
VRPP, WCVRP, and SWCVRP problem types.
"""

from .datasets import generate_datasets
from .validators import validate_gen_data_args

__all__ = ["generate_datasets", "validate_gen_data_args"]
