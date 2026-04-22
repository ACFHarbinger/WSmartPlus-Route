"""
Decoding package.

Attributes:
    DECODING_STRATEGY_REGISTRY: Registry of decoding strategies.
    get_decoding_strategy: Get decoding strategy by name.


Example:
    >>> from logic.src.utils.decoding import get_decoding_strategy
    >>> strategy = get_decoding_strategy("greedy")
    >>> strategy = get_decoding_strategy("sampling", temperature=0.7)
    >>> strategy = get_decoding_strategy("beam_search", beam_size=5)
    >>> strategy = get_decoding_strategy("evaluate", actions=torch.tensor([[0, 1], [2, 3]]))
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
from .evaluate import Evaluate
from .factory import DECODING_STRATEGY_REGISTRY, get_decoding_strategy
from .greedy import Greedy
from .sampling import Sampling

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
    "_beam_search",
    "get_beam_search_results",
]
