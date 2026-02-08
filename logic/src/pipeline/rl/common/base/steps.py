"""
Training/Validation step logic for RL4COLitModule.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Union, cast

import torch
from tensordict import TensorDict

from logic.src.utils.logging.pylogger import get_pylogger

if TYPE_CHECKING:
    from logic.src.interfaces.env import IEnv
    from logic.src.interfaces.policy import IPolicy
    from logic.src.policies.must_go import VectorizedSelector

logger = get_pylogger(__name__)


class StepMixin:
    """Mixin for training, validation, and test steps."""

    def __init__(self):
        """Initialize Class.

        Args:
            None.
        """
        # Type hints
        self.env: IEnv
        self.policy: IPolicy
        self.baseline: Any
        self.device: torch.device
        self.must_go_selector: Optional[VectorizedSelector]
        self._current_baseline_val: Any = None
        self.last_out: Any = None

    def _apply_must_go_selection(self, td: TensorDict) -> TensorDict:
        """
        Apply must-go selection to determine which bins must be collected.

        Args:
            td: TensorDict with problem instance data.

        Returns:
            TensorDict with 'must_go' mask added.
        """
        if self.must_go_selector is None:
            return td

        # Get fill levels from the TensorDict
        # WCVRP uses 'demand', VRPP uses 'prize' or 'demand'
        fill_levels = None
        for key in ["waste", "demand", "prize", "fill_level"]:
            if key in td.keys():
                fill_levels = td[key]
                break

        if fill_levels is None:
            logger.warning("No fill levels found in TensorDict for must-go selection")
            return td

        # Ensure fill_levels is 2D (batch_size, num_nodes)
        if fill_levels.dim() == 1:
            fill_levels = fill_levels.unsqueeze(0)

        # Get additional data for advanced selectors (lookahead, service_level)
        selector_kwargs = {}
        if "accumulation_rate" in td.keys():
            selector_kwargs["accumulation_rates"] = td["accumulation_rate"]
        if "std_deviation" in td.keys():
            selector_kwargs["std_deviations"] = td["std_deviation"]
        if "current_day" in td.keys():
            selector_kwargs["current_day"] = td["current_day"]

        # Get data needed for ManagerSelector (neural network-based selection)
        if "locs" in td.keys():
            selector_kwargs["locs"] = td["locs"]
        elif "loc" in td.keys():
            selector_kwargs["locs"] = td["loc"]

        # Waste history for temporal modeling
        if "waste_history" in td.keys():
            selector_kwargs["waste_history"] = td["waste_history"]
        elif "demand_history" in td.keys():
            selector_kwargs["waste_history"] = td["demand_history"]

        # Apply selector to get must-go mask
        must_go_mask = self.must_go_selector.select(fill_levels, **selector_kwargs)

        # Store in TensorDict
        td["must_go"] = must_go_mask

        return td

    @abstractmethod
    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
        env: Optional[IEnv] = None,
    ) -> torch.Tensor:
        """
        Compute RL loss.

        Args:
            td: TensorDict with environment state.
            out: Policy output dictionary.
            batch_idx: Current batch index.

        Returns:
            Loss tensor.
        """
        raise NotImplementedError

    def shared_step(
        self,
        batch: Union[TensorDict, Dict[str, Any]],
        batch_idx: int,
        phase: str,
    ) -> dict:
        """
        Common step for train/val/test.

        Args:
            batch: TensorDict batch.
            batch_idx: Batch index.
            phase: One of "train", "val", "test".

        Returns:
            Output dictionary with loss, reward, etc.
        """
        # Unwrap batch if it's from a baseline dataset
        batch, baseline_val = self.baseline.unwrap_batch(batch)

        # Move to device (crucial when pin_memory=False)
        if hasattr(batch, "to"):
            batch = cast(Any, batch).to(self.device)
        elif isinstance(batch, dict):
            batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}

        if baseline_val is not None:
            baseline_val = cast(Any, baseline_val).to(self.device)
        self._current_baseline_val = baseline_val

        # env.reset expects data on the environment's device.
        from logic.src.utils.functions.rl import ensure_tensordict

        td = ensure_tensordict(batch, self.device)

        # Apply must-go selector if configured
        if self.must_go_selector is not None:
            td = self._apply_must_go_selection(td)

        td = self.env.reset(td)

        # Run policy
        out = self.policy(
            td,
            self.env,
            strategy="sampling" if phase == "train" else "greedy",
        )

        # Compute loss for training
        if phase == "train":
            out["loss"] = self.calculate_loss(td, out, batch_idx, env=self.env)

        # Merge granular metrics from td if available
        for key in ["collection", "cost"]:
            if key in list(td.keys()):
                out[key] = td[key]

        # Log metrics
        reward_mean = out["reward"].mean()
        batch_size = out["reward"].shape[0]

        self.log(
            f"{phase}/reward",
            reward_mean,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        if "collection" in out:
            self.log(
                f"{phase}/collection",
                out["collection"].mean(),
                sync_dist=True,
                batch_size=batch_size,
            )
        if "cost" in out:
            self.log(
                f"{phase}/cost",
                out["cost"].mean(),
                sync_dist=True,
                batch_size=batch_size,
            )
        else:
            self.log(f"{phase}/cost_total", -reward_mean, sync_dist=True)

        # Store for meta-learning or logging access
        self.last_out = out

        return out

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Execute a single training step.

        Args:
            batch: Input batch, potentially wrapped by baseline.
            batch_idx: Index of the current batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        # 1. Unwrap batch if it was wrapped by baseline (e.g. RolloutBaseline)
        if hasattr(self.baseline, "unwrap_batch"):
            td, baseline_val = self.baseline.unwrap_batch(batch)
        else:
            td, baseline_val = batch, None

        # 2. Run shared step
        out = self.shared_step(td, batch_idx, phase="train")

        # 3. Calculate loss with baseline_val if available
        self._current_baseline_val = baseline_val

        return out["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> dict:
        """
        Execute a single validation step.

        Args:
            batch: TensorDict batch for validation.
            batch_idx: Index of the current batch.

        Returns:
            dict: Output dictionary with metrics.
        """
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch: Any, batch_idx: int) -> dict:
        """
        Execute a single test step.

        Args:
            batch: TensorDict batch for testing.
            batch_idx: Index of the current batch.

        Returns:
            dict: Output dictionary with metrics.
        """
        return self.shared_step(batch, batch_idx, phase="test")
