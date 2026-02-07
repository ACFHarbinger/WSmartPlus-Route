"""
Decoding package.
"""

from .base import DecodingStrategy
from .beam_search import (
    BatchBeam,
    BeamSearch,
    _beam_search,
    beam_search,
    get_beam_search_results,
)
from .decoding_utils import (
    CachedLookup,
    backtrack,
    batchify,
    gather_by_index,
    get_log_likelihood,
    modify_logits_for_top_k_filtering,
    modify_logits_for_top_p_filtering,
    segment_topk_idx,
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
    "BatchBeam",
    "beam_search",
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
    "backtrack",
    "segment_topk_idx",
    "CachedLookup",
    "DECODING_STRATEGY_REGISTRY",
]
