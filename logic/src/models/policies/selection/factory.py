"""
Factory functions for creating vectorized selectors.
"""

from typing import Optional

from logic.src.interfaces import ITraversable

from .base import VectorizedSelector
from .combined import CombinedSelector
from .last_minute import LastMinuteSelector
from .lookahead import LookaheadSelector
from .manager import ManagerSelector
from .regular import RegularSelector
from .revenue import RevenueSelector
from .service_level import ServiceLevelSelector


def create_selector_from_config(cfg) -> Optional[VectorizedSelector]:
    """
    Create a vectorized selector from a MustGoConfig or dict.

    Args:
        cfg: MustGoConfig dataclass or dict with selector configuration.
            Must have a 'strategy' field. If strategy is None, returns None.

    Returns:
        VectorizedSelector or None if no strategy specified.
    """
    if cfg is None:
        return None

    # Handle both dataclass and config-like objects (dict, DictConfig)
    if hasattr(cfg, "strategy"):
        strategy = cfg.strategy
    elif isinstance(cfg, ITraversable):
        strategy = cfg.get("strategy")
    else:
        return None

    if strategy is None:
        return None

    strategy = strategy.lower()

    if strategy == "none":
        return None

    # Extract parameters from config
    if hasattr(cfg, "__dict__"):
        params = {k: v for k, v in vars(cfg).items() if k != "strategy" and v is not None}
    elif isinstance(cfg, ITraversable):
        params = {k: v for k, v in cfg.items() if k != "strategy" and v is not None}
    else:
        params = {}

    # Handle combined strategy
    if strategy == "combined":
        combined_configs = params.get("combined_strategies", [])
        if not combined_configs:
            return None
        selectors = []
        for sub_cfg in combined_configs:
            sub_selector = create_selector_from_config(sub_cfg)
            if sub_selector is not None:
                selectors.append(sub_selector)
        if not selectors:
            return None

        logic = params.get("logic", "or")
        return CombinedSelector(selectors, logic=logic)

    # Handle manager strategy (neural network-based selection)
    if strategy == "manager":
        manager_config = {
            "hidden_dim": params.get("hidden_dim", 128),
            "lstm_hidden": params.get("lstm_hidden", 64),
            "history_length": params.get("history_length", 10),
            "critical_threshold": params.get("critical_threshold", 0.9),
        }
        device = params.get("device", "cuda")
        threshold = params.get("threshold", 0.5)
        manager_weights = params.get("manager_weights")

        selector = ManagerSelector(
            manager_config=manager_config,
            threshold=threshold,
            device=device,
        )

        # Load pre-trained weights if provided
        if manager_weights:
            selector.load_weights(manager_weights)

        return selector

    # Map strategy name to selector class and its parameters
    strategy_params = {
        "last_minute": {"threshold": params.get("threshold", 0.7)},
        "regular": {"frequency": params.get("frequency", 3)},
        "lookahead": {
            "max_fill": params.get("max_fill", 1.0),
        },
        "revenue": {
            "revenue_kg": params.get("revenue_kg", 1.0),
            "bin_capacity": params.get("bin_capacity", 1.0),
            "threshold": params.get("revenue_threshold", 0.0),
        },
        "service_level": {
            "confidence_factor": params.get("confidence_factor", 1.0),
            "max_fill": params.get("max_fill", 1.0),
        },
    }

    if strategy not in strategy_params:
        raise ValueError(
            f"Unknown strategy: {strategy}. Available: {list(strategy_params.keys()) + ['manager', 'combined']}"
        )

    return get_vectorized_selector(strategy, **strategy_params[strategy])


# Factory function for easy instantiation
def get_vectorized_selector(name: str, **kwargs) -> VectorizedSelector:
    """
    Create a vectorized selector by name.

    Args:
        name: Selector name. Options:
            - 'last_minute': Threshold-based reactive selection
            - 'regular': Periodic collection on scheduled days
            - 'lookahead': Predictive overflow-based selection
            - 'revenue': Revenue-based selection
            - 'service_level': Statistical overflow prediction
            - 'manager': Neural network-based selection (GATLSTManager)
        **kwargs: Parameters passed to the selector constructor.

    Returns:
        VectorizedSelector: The instantiated selector.

    Raises:
        ValueError: If the selector name is unknown.
    """
    selectors = {
        "last_minute": LastMinuteSelector,
        "regular": RegularSelector,
        "lookahead": LookaheadSelector,
        "revenue": RevenueSelector,
        "service_level": ServiceLevelSelector,
        "manager": ManagerSelector,
    }

    if name.lower() not in selectors:
        raise ValueError(f"Unknown selector: {name}. Available: {list(selectors.keys())}")

    return selectors[name.lower()](**kwargs)
