"""
Problem instance generators sub-package.
"""

from typing import Any

from .base import Generator
from .jssp import JSSPGenerator
from .pdp import PDPGenerator
from .scwcvrp import SCWCVRPGenerator
from .tsp import TSPGenerator
from .vrpp import VRPPGenerator
from .wcvrp import WCVRPGenerator

# Registry of available generators
GENERATOR_REGISTRY: dict[str, type[Generator]] = {
    "vrpp": VRPPGenerator,
    "cvrpp": VRPPGenerator,  # Same generator, different env handles capacity
    "wcvrp": WCVRPGenerator,
    "cwcvrp": WCVRPGenerator,
    "scwcvrp": SCWCVRPGenerator,
    "tsp": TSPGenerator,
    "pdp": PDPGenerator,
    "jssp": JSSPGenerator,
}


def get_generator(name: str, **kwargs: Any) -> Generator:
    """
    Get a generator by name.

    Args:
        name: Generator name (e.g., "vrpp", "wcvrp", "tsp").
        **kwargs: Generator configuration parameters.

    Returns:
        Configured Generator instance.

    Raises:
        ValueError: If generator name is not found.
    """
    if name not in GENERATOR_REGISTRY:
        raise ValueError(f"Unknown generator: {name}. Available: {list(GENERATOR_REGISTRY.keys())}")

    return GENERATOR_REGISTRY[name](**kwargs)


__all__ = [
    "Generator",
    "VRPPGenerator",
    "WCVRPGenerator",
    "SCWCVRPGenerator",
    "TSPGenerator",
    "PDPGenerator",
    "JSSPGenerator",
    "GENERATOR_REGISTRY",
    "get_generator",
]
