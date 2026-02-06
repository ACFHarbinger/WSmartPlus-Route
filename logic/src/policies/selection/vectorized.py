"""
Vectorized Selection Strategies for batched training.

This module provides GPU-accelerated, batched versions of the selection
strategies for use during neural network training. Each strategy operates
on tensors of shape (batch_size, num_nodes) and returns boolean masks.

The vectorized selectors determine which bins are "must go" candidates,
allowing the model to learn when to route vs when to stay at depot.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


class VectorizedSelector(ABC):
    """Abstract base class for vectorized bin selection strategies."""

    @abstractmethod
    def select(
        self,
        fill_levels: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Select bins that must be collected.

        Args:
            fill_levels: Current fill levels (batch_size, num_nodes).
                         Values in [0, 1] where 1.0 = 100% full.
            **kwargs: Strategy-specific parameters.

        Returns:
            Tensor: Boolean mask (batch_size, num_nodes) where True = must collect.
                    Note: Index 0 is the depot and should always be False.
        """
        pass


class LastMinuteSelector(VectorizedSelector):
    """
    Threshold-based reactive selection.

    Selects bins where current fill level exceeds the threshold.
    Simple but reactive - only collects when bins are nearly full.
    """

    def __init__(self, threshold: float = 0.7):
        """
        Initialize LastMinuteSelector.

        Args:
            threshold: Fill level threshold in [0, 1]. Default: 0.7 (70%).
        """
        self.threshold = threshold

    def select(
        self,
        fill_levels: Tensor,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        """
        Select bins exceeding the fill threshold.

        Args:
            fill_levels: Current fill levels (batch_size, num_nodes) in [0, 1].
            threshold: Optional override for the fill threshold.

        Returns:
            Tensor: Boolean mask (batch_size, num_nodes).
        """
        thresh = threshold if threshold is not None else self.threshold
        must_go = fill_levels > thresh

        # Depot (index 0) is never a must-go
        must_go = must_go.clone()
        must_go[:, 0] = False

        return must_go


class RegularSelector(VectorizedSelector):
    """
    Periodic collection strategy.

    Selects all bins on scheduled collection days based on a fixed frequency.
    """

    def __init__(self, frequency: int = 3):
        """
        Initialize RegularSelector.

        Args:
            frequency: Collection interval in days. Default: 3 (collect every 3rd day).
        """
        self.frequency = frequency

    def select(
        self,
        fill_levels: Tensor,
        current_day: Optional[Tensor] = None,
        frequency: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        """
        Select all bins if today is a scheduled collection day.

        Args:
            fill_levels: Current fill levels (batch_size, num_nodes).
            current_day: Current simulation day (batch_size,) or scalar.
            frequency: Optional override for collection frequency.

        Returns:
            Tensor: Boolean mask (batch_size, num_nodes).
        """
        batch_size, num_nodes = fill_levels.shape
        device = fill_levels.device
        freq = frequency if frequency is not None else self.frequency

        if freq <= 0:
            # Collect every day
            must_go = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=device)
        elif current_day is None:
            # No day info - assume collection day
            must_go = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=device)
        else:
            # Ensure current_day is a tensor
            if not isinstance(current_day, Tensor):
                current_day = torch.tensor(current_day, device=device)

            # Expand to batch if scalar
            if current_day.dim() == 0:
                current_day = current_day.expand(batch_size)

            # Collection day check: day % (freq + 1) == 1
            is_collection_day = (current_day % (freq + 1)) == 1  # (batch_size,)
            is_collection_day = is_collection_day.unsqueeze(-1)  # (batch_size, 1)

            # If collection day, all bins are must-go
            must_go = is_collection_day.expand(-1, num_nodes)

        # Depot is never a must-go
        must_go = must_go.clone()
        must_go[:, 0] = False

        return must_go


class LookaheadSelector(VectorizedSelector):
    """
    Predictive selection looking N days ahead.

    Selects bins that will overflow within the lookahead horizon based on
    current fill levels and accumulation rates.
    """

    def __init__(self, lookahead_days: int = 1, max_fill: float = 1.0):
        """
        Initialize LookaheadSelector.

        Args:
            lookahead_days: Number of days to look ahead. Default: 1.
            max_fill: Maximum fill level (overflow threshold). Default: 1.0.
        """
        self.lookahead_days = lookahead_days
        self.max_fill = max_fill

    def select(
        self,
        fill_levels: Tensor,
        accumulation_rates: Optional[Tensor] = None,
        lookahead_days: Optional[int] = None,
        max_fill: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        """
        Select bins predicted to overflow within N days.

        Args:
            fill_levels: Current fill levels (batch_size, num_nodes) in [0, 1].
            accumulation_rates: Daily fill rate (batch_size, num_nodes) in [0, 1].
            lookahead_days: Optional override for lookahead horizon.
            max_fill: Optional override for overflow threshold.

        Returns:
            Tensor: Boolean mask (batch_size, num_nodes).
        """
        horizon = lookahead_days if lookahead_days is not None else self.lookahead_days
        overflow_thresh = max_fill if max_fill is not None else self.max_fill

        if accumulation_rates is None:
            # Without rates, fall back to threshold-based selection
            must_go = fill_levels >= overflow_thresh
        else:
            # Predict future fill: current + horizon * rate
            predicted_fill = fill_levels + (horizon * accumulation_rates)
            must_go = predicted_fill >= overflow_thresh

        # Depot is never a must-go
        must_go = must_go.clone()
        must_go[:, 0] = False

        return must_go


class RevenueSelector(VectorizedSelector):
    """
    Revenue-based selection strategy.

    Selects bins where expected collection revenue exceeds a threshold.
    """

    def __init__(
        self,
        revenue_kg: float = 1.0,
        bin_capacity: float = 1.0,
        threshold: float = 0.0,
    ):
        """
        Initialize RevenueSelector.

        Args:
            revenue_kg: Revenue per kg of collected waste.
            bin_capacity: Capacity of each bin in kg.
            threshold: Minimum revenue threshold for selection.
        """
        self.revenue_kg = revenue_kg
        self.bin_capacity = bin_capacity
        self.threshold = threshold

    def select(
        self,
        fill_levels: Tensor,
        revenue_kg: Optional[float] = None,
        bin_capacity: Optional[float] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        """
        Select bins where expected revenue exceeds threshold.

        Args:
            fill_levels: Current fill levels (batch_size, num_nodes) in [0, 1].
            revenue_kg: Optional override for revenue per kg.
            bin_capacity: Optional override for bin capacity.
            threshold: Optional override for revenue threshold.

        Returns:
            Tensor: Boolean mask (batch_size, num_nodes).
        """
        rev = revenue_kg if revenue_kg is not None else self.revenue_kg
        cap = bin_capacity if bin_capacity is not None else self.bin_capacity
        thresh = threshold if threshold is not None else self.threshold

        # Expected revenue = fill_level * bin_capacity * revenue_per_kg
        expected_revenue = fill_levels * cap * rev
        must_go = expected_revenue > thresh

        # Depot is never a must-go
        must_go = must_go.clone()
        must_go[:, 0] = False

        return must_go


class ServiceLevelSelector(VectorizedSelector):
    """
    Statistical overflow prediction strategy.

    Uses mean accumulation rate and standard deviation to predict
    overflow probability and select bins accordingly.
    """

    def __init__(self, confidence_factor: float = 1.0, max_fill: float = 1.0):
        """
        Initialize ServiceLevelSelector.

        Args:
            confidence_factor: Number of standard deviations for prediction.
                              Higher = more conservative (fewer overflows).
            max_fill: Maximum fill level (overflow threshold). Default: 1.0.
        """
        self.confidence_factor = confidence_factor
        self.max_fill = max_fill

    def select(
        self,
        fill_levels: Tensor,
        accumulation_rates: Optional[Tensor] = None,
        std_deviations: Optional[Tensor] = None,
        confidence_factor: Optional[float] = None,
        max_fill: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        """
        Select bins statistically likely to overflow.

        Prediction: current + rate + (confidence * std) >= max_fill

        Args:
            fill_levels: Current fill levels (batch_size, num_nodes) in [0, 1].
            accumulation_rates: Mean daily fill rate (batch_size, num_nodes).
            std_deviations: Standard deviation of fill rate (batch_size, num_nodes).
            confidence_factor: Optional override for confidence multiplier.
            max_fill: Optional override for overflow threshold.

        Returns:
            Tensor: Boolean mask (batch_size, num_nodes).
        """
        conf = confidence_factor if confidence_factor is not None else self.confidence_factor
        overflow_thresh = max_fill if max_fill is not None else self.max_fill

        if accumulation_rates is None or std_deviations is None:
            # Without statistics, fall back to threshold-based
            must_go = fill_levels >= overflow_thresh
        else:
            # Statistical prediction: current + mean + confidence * std
            predicted_fill = fill_levels + accumulation_rates + (conf * std_deviations)
            must_go = predicted_fill >= overflow_thresh

        # Depot is never a must-go
        must_go = must_go.clone()
        must_go[:, 0] = False

        return must_go


class CombinedSelector(VectorizedSelector):
    """
    Combines multiple selection strategies with logical OR.

    A bin is selected if ANY of the constituent selectors select it.
    """

    def __init__(self, selectors: list[VectorizedSelector]):
        """
        Initialize CombinedSelector.

        Args:
            selectors: List of VectorizedSelector instances to combine.
        """
        self.selectors = selectors

    def select(
        self,
        fill_levels: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Select bins chosen by any constituent selector.

        Args:
            fill_levels: Current fill levels (batch_size, num_nodes).
            **kwargs: Passed to all constituent selectors.

        Returns:
            Tensor: Boolean mask (batch_size, num_nodes).
        """
        batch_size, num_nodes = fill_levels.shape
        device = fill_levels.device

        combined = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)

        for selector in self.selectors:
            mask = selector.select(fill_levels, **kwargs)
            combined = combined | mask

        return combined


class ManagerSelector(VectorizedSelector):
    """
    Neural network-based must-go selection using GATLSTManager.

    This selector wraps a trained HRL manager to make must-go decisions
    based on learned patterns from temporal waste data and spatial context.
    The manager learns to predict which bins require collection.
    """

    def __init__(
        self,
        manager=None,
        manager_config: Optional[dict] = None,
        threshold: float = 0.5,
        device: str = "cuda",
    ):
        """
        Initialize ManagerSelector.

        Args:
            manager: Pre-instantiated GATLSTManager. If None, creates one from config.
            manager_config: Configuration dict for creating manager if not provided.
            threshold: Probability threshold for must_go decision.
            device: Device for computation ('cpu' or 'cuda').
        """
        self.threshold = threshold
        self.device = device

        if manager is not None:
            self.manager = manager
        else:
            # Lazy import to avoid circular dependencies
            from logic.src.models.gat_lstm_manager import GATLSTManager

            config = manager_config or {}
            self.manager = GATLSTManager(
                hidden_dim=config.get("hidden_dim", 128),
                lstm_hidden=config.get("lstm_hidden", 64),
                input_dim_dynamic=config.get("history_length", 10),
                critical_threshold=config.get("critical_threshold", 0.9),
                device=device,
            )

    def select(
        self,
        fill_levels: Tensor,
        locs: Optional[Tensor] = None,
        waste_history: Optional[Tensor] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        """
        Select bins using the neural manager.

        The manager uses spatial (locations) and temporal (waste history) features
        to predict which bins must be collected.

        Args:
            fill_levels: Current fill levels (batch_size, num_nodes) in [0, 1].
            locs: Node locations (batch_size, num_nodes, 2). Required.
            waste_history: Historical waste levels (batch_size, num_nodes, history).
                          If None, expands current fill levels as history.
            threshold: Optional override for probability threshold.

        Returns:
            Tensor: Boolean mask (batch_size, num_nodes) where True = must collect.
        """
        batch_size, num_nodes = fill_levels.shape
        device = fill_levels.device
        thresh = threshold if threshold is not None else self.threshold

        # Ensure manager is on correct device
        if next(self.manager.parameters()).device != device:
            self.manager = self.manager.to(device)

        # Prepare static features (locations)
        if locs is None:
            # If no locations provided, use dummy coordinates
            # This is a fallback but locations should be provided for proper operation
            locs = torch.zeros(batch_size, num_nodes, 2, device=device)

        # Prepare dynamic features (waste history)
        if waste_history is not None:
            dynamic = waste_history
        else:
            # Expand current fill levels as constant history
            history_len = self.manager.input_dim_dynamic
            dynamic = fill_levels.unsqueeze(-1).expand(-1, -1, history_len)

        # Prepare global features
        current_waste = fill_levels
        critical_mask = (current_waste > self.manager.critical_threshold).float()
        critical_ratio = critical_mask.mean(dim=1, keepdim=True)
        max_waste = current_waste.max(dim=1, keepdim=True)[0]
        global_features = torch.cat([critical_ratio, max_waste], dim=1)

        # Get must_go mask from manager
        must_go = self.manager.get_must_go_mask(locs, dynamic, global_features, threshold=thresh)

        return must_go

    def load_weights(self, path: str):
        """Load manager weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        if "manager_state_dict" in checkpoint:
            self.manager.load_state_dict(checkpoint["manager_state_dict"])
        elif "state_dict" in checkpoint:
            self.manager.load_state_dict(checkpoint["state_dict"])
        else:
            self.manager.load_state_dict(checkpoint)


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

    # Handle both dataclass and dict
    if hasattr(cfg, "strategy"):
        strategy = cfg.strategy
    elif isinstance(cfg, dict):
        strategy = cfg.get("strategy")
    else:
        return None

    if strategy is None:
        return None

    strategy = strategy.lower()

    # Extract parameters from config
    if hasattr(cfg, "__dict__"):
        params = {k: v for k, v in vars(cfg).items() if k != "strategy" and v is not None}
    elif isinstance(cfg, dict):
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
        return CombinedSelector(selectors)

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
            "lookahead_days": params.get("lookahead_days", 1),
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
            f"Unknown strategy: {strategy}. Available: {list(strategy_params.keys())} + ['manager', 'combined']"
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
