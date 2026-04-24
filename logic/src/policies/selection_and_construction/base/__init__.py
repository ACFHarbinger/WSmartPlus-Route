"""Joint Policy Base Package.

This package defines the foundational interfaces and infrastructure for joint
selection and construction policies, including the base class and factory logic.

Attributes:
    BaseJointPolicy (Type[BaseJointPolicy]): Abstract base class for joint policies.
    JointPolicyFactory (Type[JointPolicyFactory]): Factory for instantiating policies.
    JointPolicyRegistry (Type[JointPolicyRegistry]): Shared registry for policy types.
"""

from .base_joint_policy import BaseJointPolicy
from .factory import JointPolicyFactory
from .registry import JointPolicyRegistry

__all__ = [
    "BaseJointPolicy",
    "JointPolicyRegistry",
    "JointPolicyFactory",
]
