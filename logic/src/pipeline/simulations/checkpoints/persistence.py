"""
Checkpoint file persistence management.
"""

import os
import pickle
from datetime import datetime
from typing import Any, Optional, Tuple

from loguru import logger

from logic.src.constants import ROOT_DIR


class SimulationCheckpoint:
    """
    Manages checkpoint file persistence for a single simulation run.
    """

    def __init__(self, output_dir: str, checkpoint_dir: str = "temp", policy: str = "", sample_id: int = 0):
        self.checkpoint_dir = os.path.join(ROOT_DIR, checkpoint_dir)
        self.output_dir = os.path.join(output_dir, checkpoint_dir)
        self.policy = policy
        self.sample_id = sample_id

    def get_simulation_info(self) -> dict:
        return {"policy": self.policy, "sample": self.sample_id}

    def get_checkpoint_file(self, day: Optional[int] = None, end_simulation: bool = False) -> str:
        parent_dir = self.output_dir if end_simulation else self.checkpoint_dir
        if day is not None:
            return os.path.join(parent_dir, f"checkpoint_{self.policy}_{self.sample_id}_day{day}.pkl")
        last_day = self.find_last_checkpoint_day()
        return os.path.join(parent_dir, f"checkpoint_{self.policy}_{self.sample_id}_day{last_day}.pkl")

    def save_state(self, state: Any, day: int = 0, end_simulation: bool = False) -> None:
        checkpoint_data = {
            "state": state,
            "policy": self.policy,
            "sample_id": self.sample_id,
            "day": day,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
        }

        checkpoint_file = self.get_checkpoint_file(day, end_simulation)
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)

    def load_state(self, day: Optional[int] = None) -> Tuple[Optional[Any], int]:
        checkpoint_files = []

        if day is not None:
            checkpoint_files.append(self.get_checkpoint_file(day))

        checkpoint_files.append(self.get_checkpoint_file(self.find_last_checkpoint_day()))

        for checkpoint_file in checkpoint_files:
            if os.path.exists(checkpoint_file):
                try:
                    with open(checkpoint_file, "rb") as f:
                        checkpoint_data = pickle.load(f)

                    if (
                        checkpoint_data.get("policy") == self.policy
                        and checkpoint_data.get("sample_id") == self.sample_id
                    ):
                        return checkpoint_data["state"], checkpoint_data.get("day", 0)
                    else:
                        logger.warning(f"Checkpoint mismatch: expected {self.policy}_{self.sample_id}")
                except Exception as e:
                    logger.error(f"Error loading checkpoint {checkpoint_file}: {e}")
        logger.warning("No valid checkpoint found")
        return None, 0

    def find_last_checkpoint_day(self) -> int:
        max_day = 0
        pattern = f"checkpoint_{self.policy}_{self.sample_id}_day"
        if not os.path.exists(self.checkpoint_dir):
            return 0
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(pattern) and filename.endswith(".pkl"):
                try:
                    day_num = int(filename.split("day")[1].split(".pkl")[0])
                    max_day = max(max_day, day_num)
                except (ValueError, IndexError):
                    continue
        return max_day

    def clear(self, policy: Optional[str] = None, sample_id: Optional[int] = None) -> int:
        if policy is None:
            policy = self.policy
        if sample_id is None:
            sample_id = self.sample_id

        pattern = f"checkpoint_{policy}_{sample_id}"
        removed_count = 0
        if not os.path.exists(self.checkpoint_dir):
            return 0
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(pattern) and filename.endswith(".pkl"):
                try:
                    os.remove(os.path.join(self.checkpoint_dir, filename))
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Error removing {filename}: {e}")
        return removed_count

    def delete_checkpoint_day(self, day: int) -> bool:
        checkpoint_file = self.get_checkpoint_file(day)
        try:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing {checkpoint_file}: {e}")
            return False
