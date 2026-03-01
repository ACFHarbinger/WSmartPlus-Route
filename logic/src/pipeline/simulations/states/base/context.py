"""context.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import context
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from logic.src.constants import (
    ROOT_DIR,
    TQDM_COLOURS,
)

from .base import SimState

if TYPE_CHECKING:
    from logic.src.configs import Config

    from ...checkpoints import SimulationCheckpoint


class SimulationContext:
    """
    Context object for the Simulation State Machine.

    Stores the simulation configuration and runtime state. All simulation
    states access configuration via ``self.cfg.sim.*`` (typed) instead of
    the legacy ``self.args[...]`` dict pattern.
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

    pol_name: str = ""
    pol_id: int = 0
    pol_cfg: Dict[str, Any] = {}

    @property
    def policy(self) -> str:
        """Alias for pol_name for backward compatibility."""
        return self.pol_name

    def __init__(
        self,
        cfg: Union[Config, DictConfig],
        device: torch.device,
        indices: List[int],
        sample_id: int,
        pol_id: int,
        model_weights_path: Optional[str],
        variables_dict: Dict[str, Any],
    ):
        """Initialize Class.

        Args:
            cfg (Config): Typed Hydra configuration.
            device (torch.device): Description of device.
            indices (List[int]): Description of indices.
            sample_id (int): Description of sample_id.
            pol_id (int): Description of pol_id.
            model_weights_path (Optional[str]): Description of model_weights_path.
            variables_dict (Dict[str, Any]): Description of variables_dict.
        """
        self.cfg = cfg
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

        sim = cfg.sim
        self.data_dir = os.path.join(ROOT_DIR, "data", "wsr_simulator")
        self.results_dir = os.path.join(
            ROOT_DIR,
            "assets",
            sim.output_dir,
            str(sim.days) + "_days",
            str(sim.graph.area) + "_" + str(sim.graph.num_loc),
        )
        policies = sim.full_policies
        raw_policy = policies[pol_id]
        if isinstance(raw_policy, dict) and len(raw_policy) == 1:
            self.pol_name = list(raw_policy.keys())[0]
            self.pol_cfg = raw_policy[self.pol_name]
        elif isinstance(raw_policy, dict):
            self.pol_name = "unknown"
            self.pol_cfg = raw_policy
        else:
            self.pol_name = str(raw_policy)
            self.pol_cfg = {}

        self._continue_init(variables_dict, pol_id)

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
