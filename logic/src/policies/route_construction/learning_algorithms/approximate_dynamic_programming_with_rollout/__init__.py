"""
Module documentation.

Attributes:
    ADPRolloutPolicy: Approximate Dynamic Programming with Rollout Policy.

Example:
    >>> from logic.src.policies.route_construction.learning_algorithms import ADPRolloutPolicy
    >>> policy = ADPRolloutPolicy()
    >>> policy.construct_routes(env)
"""

from .policy_adp import ADPRolloutPolicy

__all__ = ["ADPRolloutPolicy"]
