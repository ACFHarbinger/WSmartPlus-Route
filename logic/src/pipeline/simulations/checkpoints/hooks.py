"""
Checkpoint hooks for simulation lifecycle.
"""

import time
import traceback
from typing import Any, Callable, Dict, Optional

from loguru import logger


class CheckpointHook:
    """
    Orchestrates checkpoint timing and error handling during simulation.
    """

    def __init__(self, checkpoint, checkpoint_interval: int, state_getter: Optional[Callable[[], Any]] = None):
        self.checkpoint = checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.day = 0
        self.tic: Optional[float] = None
        self.state_getter = state_getter

    def get_current_day(self) -> int:
        return self.day

    def get_checkpoint_info(self) -> dict:
        return {"checkpoint": self.checkpoint, "interval": self.checkpoint_interval}

    def set_timer(self, tic: float) -> None:
        self.tic = tic

    def set_state_getter(self, state_getter: Callable[[], Any]) -> None:
        self.state_getter = state_getter

    def before_day(self, day: int) -> None:
        self.day = day

    def after_day(self, tic: Optional[float] = None, delete_previous: bool = False) -> None:
        previous_checkpoint_day = self.checkpoint.find_last_checkpoint_day()
        if tic:
            self.tic = tic
        if (
            self.checkpoint
            and self.checkpoint_interval > 0
            and self.day % self.checkpoint_interval == 0
            and self.state_getter
        ):
            state_snapshot = self.state_getter()
            self.checkpoint.save_state(state_snapshot, self.day)
        if delete_previous:
            self.checkpoint.delete_checkpoint_day(previous_checkpoint_day)

    def on_error(self, error: Exception) -> Dict[str, Any]:
        execution_time = time.process_time() - self.tic if self.tic else 0
        day = self.get_current_day()
        info = self.checkpoint.get_simulation_info()
        policy = info.get("policy", "")
        sample_id = info.get("sample", 0)
        logger.error(f"Crash in {policy} #{sample_id} at day {day}: {error}")

        traceback.print_exc()
        if self.checkpoint and self.state_getter:
            try:
                state_snapshot = self.state_getter()
                self.checkpoint.save_state(state_snapshot, self.day)
            except Exception as save_error:
                logger.error(f"Failed to save emergency checkpoint: {save_error}")

        return {
            "policy": policy,
            "sample_id": sample_id,
            "day": self.day,
            "error": str(error),
            "error_type": type(error).__name__,
            "execution_time": execution_time,
            "success": False,
        }

    def on_completion(self, policy: Optional[str] = None, sample_id: Optional[int] = None) -> None:
        if self.checkpoint:
            self.checkpoint.clear(policy, sample_id)
        if self.state_getter:
            state_snapshot = self.state_getter()
            self.checkpoint.save_state(state_snapshot, self.day, end_simulation=True)
