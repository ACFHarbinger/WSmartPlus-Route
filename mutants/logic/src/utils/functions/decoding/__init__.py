"""
Decoding package.
"""

from .base import DecodingStrategy
from .beam_search import BeamSearch
from .decoding_utils import (
    batchify,
    gather_by_index,
    get_log_likelihood,
    modify_logits_for_top_k_filtering,
    modify_logits_for_top_p_filtering,
    top_k_filter,
    top_p_filter,
    unbatchify,
)
from .strategies import Evaluate, Greedy, Sampling

DECODING_STRATEGY_REGISTRY = {
    "greedy": Greedy,
    "sampling": Sampling,
    "beam_search": BeamSearch,
    "evaluate": Evaluate,
}


def get_decoding_strategy(
    name: str,
    **kwargs,
) -> DecodingStrategy:
    """
    Get decoding strategy by name.

    Args:
        name: Strategy name ('greedy', 'sampling', 'beam_search', 'evaluate')
        **kwargs: Strategy-specific parameters

    Returns:
        Initialized decoding strategy
    """
    name = name.lower()
    if name not in DECODING_STRATEGY_REGISTRY:
        raise ValueError(f"Unknown decoding strategy: {name}. Available: {list(DECODING_STRATEGY_REGISTRY.keys())}")
    return DECODING_STRATEGY_REGISTRY[name](**kwargs)


__all__ = [
    "DecodingStrategy",
    "Greedy",
    "Sampling",
    "BeamSearch",
    "Evaluate",
    "get_decoding_strategy",
    "get_log_likelihood",
    "modify_logits_for_top_k_filtering",
    "modify_logits_for_top_p_filtering",
    "top_k_filter",
    "top_p_filter",
    "batchify",
    "unbatchify",
    "gather_by_index",
    "DECODING_STRATEGY_REGISTRY",
]
