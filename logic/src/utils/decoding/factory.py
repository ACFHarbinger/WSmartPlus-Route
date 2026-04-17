from .base import DecodingStrategy
from .beam_search import BeamSearch
from .evaluate import Evaluate
from .greedy import Greedy
from .sampling import Sampling

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
