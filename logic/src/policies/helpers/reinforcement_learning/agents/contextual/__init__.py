"""Contextual Bandits Module.

Provides agents that utilize environmental context (features) to inform
the selection of operations or parameters.

Attributes:
    ContextualBanditAgent: Base class for all contextual agents.
    LinUCBAgent: Linear Upper Confidence Bound implementation.
    ContextualThompsonSamplingAgent: Bayesian linear regression approach.
    GPCMABAgent: Gaussian Process-based combinatorial bandit.

Example:
    >>> from logic.src.policies.helpers.reinforcement_learning.agents.contextual import LinUCBAgent
    >>> agent = LinUCBAgent(n_arms=3, feature_dim=10)
"""

from .base import ContextualBanditAgent
from .gpcmab import GPCMABAgent
from .linucb import LinUCBAgent
from .thompson import ContextualThompsonSamplingAgent

__all__ = [
    "ContextualBanditAgent",
    "LinUCBAgent",
    "ContextualThompsonSamplingAgent",
    "GPCMABAgent",
]
