"""
Evaluation Dispatcher.
"""

from typing import Any, Dict

from logic.src.pipeline.features.evaluators import GreedyEval, SamplingEval
from torch.utils.data import DataLoader


def evaluate_policy(
    policy: Any, env: Any, data_loader: DataLoader, method: str = "greedy", **kwargs
) -> Dict[str, float]:
    """
    Evaluate a policy using the specified method.

    Args:
        policy: Policy to evaluate
        env: Environment
        data_loader: Dataset loader
        method: Decoding strategy ("greedy", "sampling", "bs", etc.)
        **kwargs: Additional arguments for evaluator

    Returns:
        Dict with metrics
    """
    if method == "greedy":
        evaluator = GreedyEval(env, **kwargs)
    elif method == "sampling":
        evaluator = SamplingEval(env, **kwargs)
    else:
        # Fallback or error? For now default to greedy if unknown or raise
        raise ValueError(f"Unknown evaluation method: {method}")

    return evaluator(policy, data_loader)
