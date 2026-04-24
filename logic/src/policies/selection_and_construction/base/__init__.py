"""Joint Policy Base Package.

This package defines the foundational interfaces and infrastructure for joint
selection and construction policies, including the base class and factory logic.

Attributes:
    BaseJointPolicy: Abstract base class for joint policies.
    JointPolicyFactory: Factory for instantiating policies.
    JointPolicyRegistry: Shared registry for policy types.

Example:
    >>> from logic.src.policies.selection_and_construction.base import JointPolicyFactory
    >>> policy = JointPolicyFactory.create("jgo")
"""

from .base_joint_policy import BaseJointPolicy
from .factory import JointPolicyFactory
from .registry import JointPolicyRegistry

__all__ = [
    "BaseJointPolicy",
    "JointPolicyRegistry",
    "JointPolicyFactory",
]
