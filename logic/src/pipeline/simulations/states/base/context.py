"""context.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import context
"""

from __future__ import annotations

import contextlib
import os
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from logic.src.constants import (
    CONFIG_CHAR_POLICIES,
    ENGINE_POLICIES,
    ROOT_DIR,
    SIMPLE_POLICIES,
    THRESHOLD_POLICIES,
    TQDM_COLOURS,
)

from .base import SimState

if TYPE_CHECKING:
    from ...checkpoints import SimulationCheckpoint


class SimulationContext:
    """
    Context object for the Simulation State Machine.
    """

    lock: Optional[threading.Lock]
    counter: Optional[Any]
    overall_progress: Optional[Any]
    shared_metrics: Any = None
    pbar: Optional[Any]
    log_path: str = ""
    exec_time: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    pol_name: Optional[str] = None
    pol_engine: Optional[str] = None
    pol_threshold: Optional[float] = None
    pol_id: int = 0
    pol_strip: str = ""

    def __init__(
        self,
        opts: Dict[str, Any],
        device: torch.device,
        indices: List[int],
        sample_id: int,
        pol_id: int,
        model_weights_path: str,
        variables_dict: Dict[str, Any],
    ):
        """Initialize Class.

        Args:
            opts (Dict[str, Any]): Description of opts.
            device (torch.device): Description of device.
            indices (List[int]): Description of indices.
            sample_id (int): Description of sample_id.
            pol_id (int): Description of pol_id.
            model_weights_path (str): Description of model_weights_path.
            variables_dict (Dict[str, Any]): Description of variables_dict.
        """
        self.opts = opts
        self.device = device
        self.indices = indices
        self.sample_id = sample_id
        self.pol_id = pol_id
        self.model_weights_path = model_weights_path

        self.lock = variables_dict.get("lock")
        self.counter = variables_dict.get("counter")
        self.overall_progress = variables_dict.get("overall_progress")
        self.shared_metrics = variables_dict.get("shared_metrics")

        self.current_state: Optional[SimState] = None
        self.result: Optional[Dict[str, Any]] = None

        self.data_dir = os.path.join(ROOT_DIR, "data", "wsr_simulator")
        self.results_dir = os.path.join(
            ROOT_DIR,
            "assets",
            opts["output_dir"],
            str(opts["days"]) + "_days",
            str(opts["area"]) + "_" + str(opts["size"]),
        )
        self.policy = opts["policies"][pol_id]
        self.pol_strip, self.data_dist = self.policy.rsplit("_", 1)

        self.pol_name = ""
        self.pol_engine = None
        self.pol_threshold = None
        self._parse_policy_string()
        self._continue_init(variables_dict, pol_id)

    def _parse_policy_string(self) -> None:
        """parse policy string."""
        parts = self.pol_strip.split("_")
        for pol_key, engines in ENGINE_POLICIES.items():
            if pol_key in parts:
                self.pol_name = pol_key
                for eng in engines:
                    if eng in parts:
                        self.pol_engine = eng
                        break
                self._extract_threshold(pol_key)
                return

        for pol_key, chars in CONFIG_CHAR_POLICIES.items():
            if pol_key in parts:
                self.pol_name = pol_key
                self._extract_threshold_with_config_char(pol_key, chars)
                return

        for pol_key in THRESHOLD_POLICIES:
            if pol_key in parts:
                self.pol_name = pol_key
                self._extract_threshold(pol_key)
                return

        for keywords, name in SIMPLE_POLICIES.items():
            if any(kw in parts or self.pol_strip.startswith(kw) for kw in keywords):
                self.pol_name = name
                return

        self.pol_name = self.pol_strip

    def _extract_threshold(self, policy_key: str) -> None:
        """extract threshold.

        Args:
            policy_key (str): Description of policy_key.
        """
        try:
            parts = self.pol_strip.split(policy_key)
            if len(parts) > 1:
                threshold_part = parts[1].strip("_")
                sub_parts = threshold_part.split("_")
                if sub_parts[0]:
                    self.pol_threshold = float(sub_parts[0])
        except (ValueError, IndexError):
            pass

    def _extract_threshold_with_config_char(self, policy_key: str, config_chars: List[str]) -> None:
        """extract threshold with config char.

        Args:
            policy_key (str): Description of policy_key.
            config_chars (List[str]): Description of config_chars.
        """
        try:
            parts = self.pol_strip.split(policy_key)
            if len(parts) > 1:
                threshold_part = parts[1].strip("_")
                sub_parts = threshold_part.split("_")
                if sub_parts[0] in config_chars and len(sub_parts) > 1:
                    self.pol_threshold = float(sub_parts[1])
                elif sub_parts[0]:
                    with contextlib.suppress(ValueError):
                        self.pol_threshold = float(sub_parts[0])
        except (ValueError, IndexError):
            pass

    def _continue_init(self, variables_dict: Dict[str, Any], pol_id: int) -> None:
        """continue init.

        Args:
            variables_dict (Dict[str, Any]): Description of variables_dict.
            pol_id (int): Description of pol_id.
        """
        self.start_day: int = 1
        self.checkpoint: Optional[SimulationCheckpoint] = None
        self.bins: Optional[Any] = None
        self.new_data: Optional[pd.DataFrame] = None
        self.coords: Optional[pd.DataFrame] = None
        self.dist_tup: Optional[Tuple[np.ndarray, Any, Any, Any]] = None
        self.model_tup: Optional[Tuple[Any, ...]] = None
        self.model_env: Optional[Any] = None
        self.hrl_manager: Optional[Any] = None
        self.cached: Optional[List[Any]] = None
        self.run_time: float = 0
        self.overflows: int = 0
        self.current_collection_day: int = 0
        self.daily_log: Optional[Dict[str, List[Any]]] = None
        self.attention_dict: Dict[str, List[Any]] = {}
        self.execution_time: float = 0
        self.tic: float = 0
        self.config: Optional[Dict[str, Any]] = None
        self.vehicle_capacity: Optional[float] = None

        self.tqdm_pos: int = variables_dict.get("tqdm_pos", 0)
        self.colour: str = TQDM_COLOURS[pol_id % len(TQDM_COLOURS)]

        # Early import to avoid circular dependency
        from ..initializing import InitializingState

        self.transition_to(InitializingState())

    def transition_to(self, state: Optional[SimState]) -> None:
        """Transition to.

        Args:
            state (Optional[SimState]): Description of state.
        """
        self.current_state = state
        if self.current_state is not None:
            self.current_state.context = self

    def run(self) -> Optional[Dict[str, Any]]:
        """Run.

        Returns:
            Any: Description.
        """
        while self.current_state is not None:
            self.current_state.handle(self)
        return self.result

    def get_current_state_tuple(self) -> Tuple[Any, ...]:
        """Get current state tuple.

        Returns:
            Any: Description.
        """
        return (
            self.new_data,
            self.coords,
            self.dist_tup,
            None,
            self.bins,
            self.model_tup,
            self.cached,
            self.overflows,
            self.current_collection_day,
            self.daily_log,
            self.execution_time,
        )
