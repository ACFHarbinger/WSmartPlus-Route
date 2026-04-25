"""Contextual Multi-Armed Bandit (CMAB) agents.

Provides linear and non-linear agents for operator selection and parameter
control based on environmental context.

Attributes:
    ContextualBanditAgent: Base interface for CMAB agents.
    LinUCBAgent: Linear Upper Confidence Bound implementation.
    ContextualThompsonSamplingAgent: Bayesian linear regression agent.
    GPCMABAgent: Gaussian Process-based combinatorial agent.

Example:
    >>> from logic.src.policies.helpers.reinforcement_learning.agents.contextual_bandits import LinUCBAgent
    >>> agent = LinUCBAgent(n_arms=5, feature_dim=10)
"""

from .contextual.base import ContextualBanditAgent
from .contextual.gpcmab import GPCMABAgent
from .contextual.linucb import LinUCBAgent
from .contextual.thompson import ContextualThompsonSamplingAgent

__all__ = [
    "ContextualBanditAgent",
    "LinUCBAgent",
    "ContextualThompsonSamplingAgent",
    "GPCMABAgent",
]
