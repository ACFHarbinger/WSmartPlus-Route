"""
Enums and Registry Module.

This package centralizes all tags and the global registry used for algorithm
classification, discovery, and orchestration across the WSmart-Route framework.

Attributes:
    GlobalRegistry: Global registry for algorithm classification, discovery, and orchestration
    PolicyTag: Policy tags
    OperatorTag: Operator tags
    EnvironmentTag: Environment tags
    ModelTag: Model tags
    TrainerTag: Trainer tags

Example:
    >>> from logic.src.enums import (
        GlobalRegistry,
        PolicyTag,
        OperatorTag,
        EnvironmentTag,
        ModelTag,
        TrainerTag,
    )
"""

from .environment_tags import EnvironmentTag
from .model_tags import ModelTag
from .operator_tags import OperatorTag
from .policy_tags import PolicyTag
from .registry import GlobalRegistry
from .trainer_tags import TrainerTag

__all__ = [
    "GlobalRegistry",
    "PolicyTag",
    "OperatorTag",
    "EnvironmentTag",
    "ModelTag",
    "TrainerTag",
]
