"""
Environment module for WSmart-Route.

This module provides RL4CO-style environment abstractions for
combinatorial optimization problems.
"""

from logic.src.envs.base import ImprovementEnvBase, RL4COEnvBase
from logic.src.envs.generators import (
    GENERATOR_REGISTRY,
    Generator,
    VRPPGenerator,
    WCVRPGenerator,
    get_generator,
)
from logic.src.envs.vrpp import CVRPPEnv, VRPPEnv
from logic.src.envs.wcvrp import CWCVRPEnv, SDWCVRPEnv, WCVRPEnv

# Environment registry
ENV_REGISTRY = {
    "vrpp": VRPPEnv,
    "cvrpp": CVRPPEnv,
    "wcvrp": WCVRPEnv,
    "cwcvrp": CWCVRPEnv,
    "sdwcvrp": SDWCVRPEnv,
}


def get_env(name: str, **kwargs) -> RL4COEnvBase:
    """
    Factory function to get environment by name.

    Args:
        name: Environment name (vrpp, cvrpp, wcvrp, etc.)
        **kwargs: Environment configuration parameters.

    Returns:
        Initialized environment instance.

    Raises:
        ValueError: If environment name is not recognized.
    """
    name = name.lower()
    if name not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment: {name}. Available: {list(ENV_REGISTRY.keys())}")
    return ENV_REGISTRY[name](**kwargs)


__all__ = [
    # Base classes
    "RL4COEnvBase",
    "ImprovementEnvBase",
    # Generators
    "Generator",
    "VRPPGenerator",
    "WCVRPGenerator",
    "get_generator",
    "GENERATOR_REGISTRY",
    # Environments
    "VRPPEnv",
    "CVRPPEnv",
    "WCVRPEnv",
    "CWCVRPEnv",
    "SDWCVRPEnv",
    # Registry
    "ENV_REGISTRY",
    "get_env",
]
