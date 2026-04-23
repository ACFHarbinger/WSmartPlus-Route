"""Selection strategy factory.

This module provides factory functions to create vectorized bin selection
strategies from configuration objects or by explicit name.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

from logic.src.interfaces import ITraversable

from .base import VectorizedSelector
from .combined import CombinedSelector
from .last_minute import LastMinuteSelector
from .lookahead import LookaheadSelector
from .manager import ManagerSelector
from .regular import RegularSelector
from .revenue import RevenueSelector
from .service_level import ServiceLevelSelector


def create_selector_from_config(cfg: Any) -> Optional[VectorizedSelector]:
    """Create a vectorized selector from a MandatorySelectionConfig or dict.

    Args:
        cfg: Configuration object containing 'strategy' and 'params'.

    Returns:
        Optional[VectorizedSelector]: The instantiated selector or None.
    """
    if cfg is None:
        return None

    strategy = _get_strategy(cfg)
    if not strategy or strategy.lower() == "none":
        return None

    strategy = strategy.lower()
    params = _get_params(cfg)

    if strategy == "combined":
        return _create_combined_selector(params)
    if strategy == "manager":
        return _create_manager_selector(params)

    strategy_params = _get_strategy_params(strategy, params)
    return get_vectorized_selector(strategy, **strategy_params)


def _get_strategy(cfg: object) -> Optional[str]:
    """Extract strategy name from config.

    Args:
        cfg: The configuration object.

    Returns:
        Optional[str]: Strategy name if found, else None.
    """
    if hasattr(cfg, "strategy"):
        return str(cfg.strategy)
    if isinstance(cfg, ITraversable):
        return cast(Optional[str], cfg.get("strategy"))
    return None


def _get_params(cfg: object) -> Dict[str, Any]:
    """Extract parameters from config, excluding strategy.

    Args:
        cfg: The configuration object.

    Returns:
        Dict[str, Any]: Dictionary of strategy parameters.
    """
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in vars(cfg).items() if k != "strategy" and v is not None}
    if isinstance(cfg, ITraversable):
        return {k: v for k, v in cfg.items() if k != "strategy" and v is not None}
    return {}


def _create_combined_selector(params: Dict[str, Any]) -> Optional[VectorizedSelector]:
    """Helper to create a CombinedSelector.

    Args:
        params: Combined selector parameters.

    Returns:
        Optional[VectorizedSelector]: Instantiated CombinedSelector or None.
    """
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
    return CombinedSelector(selectors, logic=params.get("logic", "or"))


def _create_manager_selector(params: Dict[str, Any]) -> VectorizedSelector:
    """Helper to create a ManagerSelector.

    Args:
        params: Neural manager parameters.

    Returns:
        VectorizedSelector: Instantiated ManagerSelector.
    """
    manager_config = {
        "hidden_dim": params.get("hidden_dim", 128),
        "lstm_hidden": params.get("lstm_hidden", 64),
        "history_length": params.get("history_length", 10),
        "critical_threshold": params.get("critical_threshold", 0.9),
    }
    selector = ManagerSelector(
        manager_config=manager_config,
        threshold=params.get("threshold", 0.5),
        device=params.get("device", "cuda"),
    )
    if params.get("manager_weights"):
        selector.load_weights(params["manager_weights"])
    return selector


def _get_strategy_params(strategy: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Map strategy name to its specific parameters.

    Args:
        strategy: name of the strategy.
        params: raw parameter dictionary.

    Returns:
        Dict[str, Any]: mapped parameters for the selector constructor.

    Raises:
        ValueError: if strategy is unknown.
    """
    mappings = {
        "last_minute": {"threshold": params.get("threshold", 0.7)},
        "regular": {"frequency": params.get("frequency", 3)},
        "lookahead": {"max_fill": params.get("max_fill", 1.0)},
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
    if strategy not in mappings:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(mappings.keys())} + manager, combined")
    return mappings[strategy]


def get_vectorized_selector(name: str, **kwargs: Any) -> VectorizedSelector:
    """Create a vectorized selector by name.

    Args:
        name: Selector name. Options:
            - 'last_minute': Threshold-based reactive selection.
            - 'regular': Periodic collection on scheduled days.
            - 'lookahead': Predictive overflow-based selection.
            - 'revenue': Revenue-based selection.
            - 'service_level': Statistical overflow prediction.
            - 'manager': Neural network-based selection (MandatoryManager).
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

    selector_cls = cast(Any, selectors[name.lower()])
    return cast(VectorizedSelector, selector_cls(**kwargs))
