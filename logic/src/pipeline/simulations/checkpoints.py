"""
Checkpointing System for Long-Running Simulations.

This module implements fault-tolerant state persistence for multi-day simulations.
It provides automatic checkpointing, crash recovery, and resumption capabilities
to prevent data loss in long experiments.

Architecture:
    - SimulationCheckpoint: Manages file I/O for checkpoint persistence
    - CheckpointHook: Coordinates checkpoint timing and error handling
    - checkpoint_manager: Context manager for automatic lifecycle management
    - CheckpointError: Custom exception for graceful error propagation

Features:
    - Periodic state snapshots (every N days)
    - Emergency checkpoints on crashes
    - Resume from last valid checkpoint
    - Per-policy, per-sample isolation
    - Metadata tracking (timestamp, version)

Usage:
    with checkpoint_manager(checkpoint, interval, state_getter) as hook:
        for day in range(n_days):
            hook.before_day(day)
            run_simulation_day()
            hook.after_day()

Classes:
    SimulationCheckpoint: Checkpoint storage and retrieval
    CheckpointHook: Lifecycle event handler for checkpointing
    CheckpointError: Exception wrapper for error results
"""

import os
import pickle
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict

from logic.src.constants import ROOT_DIR


class SimulationCheckpoint:
    """
    Manages checkpoint file persistence for a single simulation run.

    Handles saving and loading simulation state snapshots to disk.
    Each checkpoint is uniquely identified by (policy, sample_id, day).

    Checkpoint files contain:
        - Full simulation state (bins, data, models, etc.)
        - Metadata (policy, sample_id, day, timestamp, version)

    Attributes:
        checkpoint_dir: Temporary checkpoint storage (auto-cleaned)
        output_dir: Final checkpoint destination (preserved)
        policy: Policy identifier string
        sample_id: Sample/seed identifier
    """

    def __init__(self, output_dir, checkpoint_dir="temp", policy="", sample_id=0):
        """
        Initializes checkpoint manager for a specific simulation run.

        Args:
            output_dir: Directory for final output files
            checkpoint_dir: Temporary directory name (default: 'temp')
            policy: Policy identifier (e.g., 'am_gat_emp')
            sample_id: Sample/seed number for this run
        """
        self.checkpoint_dir = os.path.join(ROOT_DIR, checkpoint_dir)
        self.output_dir = os.path.join(output_dir, checkpoint_dir)
        self.policy = policy
        self.sample_id = sample_id

    def get_simulation_info(self):
        """
        Returns a summary of the current simulation run.

        Returns:
            Dict: Mapping of policy name and sample identifier.
        """
        return {"policy": self.policy, "sample": self.sample_id}

    def get_checkpoint_file(self, day=None, end_simulation=False):
        """
        Generates the absolute path for a checkpoint file.

        Args:
            day: Simulation day for the checkpoint (None = latest).
            end_simulation: Whether this is the final simulation result.

        Returns:
            str: Path to the .pkl checkpoint file.
        """
        parent_dir = self.output_dir if end_simulation else self.checkpoint_dir
        if day is not None:
            return os.path.join(parent_dir, f"checkpoint_{self.policy}_{self.sample_id}_day{day}.pkl")
        last_day = self.find_last_checkpoint_day()
        return os.path.join(parent_dir, f"checkpoint_{self.policy}_{self.sample_id}_day{last_day}.pkl")

    def save_state(self, state, day=0, end_simulation=False):
        """
        Serializes and saves the current simulation state to disk.

        Args:
            state: The simulation state object to save.
            day: The simulation day for this snapshot.
            end_simulation: Whether to save to final results or temp storage.
        """
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

    def load_state(self, day=None):
        """
        Loads a simulation state snapshot from disk.

        Args:
            day: Specific day to load (None = automatically find latest).

        Returns:
            Tuple[Any, int]: (State object, day number) or (None, 0) if not found.
        """
        checkpoint_files = []

        # Try specific day first
        if day is not None:
            specific_file = self.get_checkpoint_file(day)
            checkpoint_files.append(specific_file)

        # Always try latest
        latest_file = self.get_checkpoint_file(self.find_last_checkpoint_day())
        checkpoint_files.append(latest_file)
        for checkpoint_file in checkpoint_files:
            if os.path.exists(checkpoint_file):
                try:
                    with open(checkpoint_file, "rb") as f:
                        checkpoint_data = pickle.load(f)

                    # Verify this checkpoint matches our policy/sample
                    if (
                        checkpoint_data.get("policy") == self.policy
                        and checkpoint_data.get("sample_id") == self.sample_id
                    ):
                        return checkpoint_data["state"], checkpoint_data.get("day", 0)
                    else:
                        print(f"Checkpoint mismatch: expected {self.policy}_{self.sample_id}")
                except Exception as e:
                    print(f"Error loading checkpoint {checkpoint_file}: {e}")
        print("Warning: no valid checkpoint found")
        return None, 0

    def find_last_checkpoint_day(self):
        """
        Identifies the highest day number for which a checkpoint file exists.

        Returns:
            int: The maximum day number found, or 0 if no checkpoints exist.
        """
        max_day = 0
        pattern = f"checkpoint_{self.policy}_{self.sample_id}_day"
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(pattern) and filename.endswith(".pkl"):
                try:
                    day_num = int(filename.split("day")[1].split(".pkl")[0])
                    max_day = max(max_day, day_num)
                except ValueError:
                    continue
        return max_day

    def clear(self, policy=None, sample_id=None):
        """
        Deletes all checkpoint files matching the specified criteria.

        Args:
            policy: Policy name to clear (None = self.policy).
            sample_id: Sample identifier to clear (None = self.sample_id).

        Returns:
            int: The number of files successfully removed.
        """
        if policy is None:
            policy = self.policy
        if sample_id is None:
            sample_id = self.sample_id

        pattern = f"checkpoint_{policy}_{sample_id}"
        removed_count = 0
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(pattern) and filename.endswith(".pkl"):
                try:
                    os.remove(os.path.join(self.checkpoint_dir, filename))
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {filename}: {e}")
        return removed_count

    def delete_checkpoint_day(self, day):
        """
        Deletes a specific daily checkpoint file.

        Args:
            day: The simulation day of the checkpoint to delete.

        Returns:
            bool: True if deleted successfully, False otherwise.
        """
        checkpoint_file = self.get_checkpoint_file(day)
        try:
            os.remove(os.path.join(self.checkpoint_dir, checkpoint_file))
            return True
        except Exception as e:
            print(f"Error removing {checkpoint_file}: {e}")
            return False


class CheckpointHook:
    """
    Orchestrates checkpoint timing and error handling during simulation.

    Provides lifecycle hooks (before_day, after_day, on_error, on_completion)
    to automatically checkpoint state at regular intervals and on failures.

    The hook pattern decouples checkpointing logic from simulation logic,
    enabling reusable, testable, and composable checkpoint strategies.

    Attributes:
        checkpoint: SimulationCheckpoint instance for file I/O
        checkpoint_interval: Save every N days (0 = disabled)
        day: Current simulation day
        tic: Timer reference for execution time tracking
        state_getter: Callable that returns current state snapshot
    """

    def __init__(self, checkpoint, checkpoint_interval, state_getter=None):
        """
        Initializes checkpoint hook with timing and state access.

        Args:
            checkpoint: SimulationCheckpoint for persistence
            checkpoint_interval: Days between checkpoints (0 = disabled)
            state_getter: Callable[[], Any] that returns state snapshot
        """
        self.checkpoint = checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.day = 0
        self.tic = None
        self.state_getter = state_getter

    def get_current_day(self):
        """
        Retrieves the simulation day currently being processed.

        Returns:
            int: The current simulation day.
        """
        return self.day

    def get_checkpoint_info(self):
        """
        Returns the checkpointing configuration.

        Returns:
            Dict: Mapping of checkpoint manager and save interval.
        """
        return {"checkpoint": self.checkpoint, "interval": self.checkpoint_interval}

    def set_timer(self, tic):
        """
        Sets the reference point for execution time calculations.

        Args:
            tic: Value from time.process_time().
        """
        self.tic = tic

    def set_state_getter(self, state_getter: Callable[[], Any]):
        """
        Sets the callback used to capture the current simulation state.

        Args:
            state_getter: Function that returns a state snapshot.
        """
        self.state_getter = state_getter

    def before_day(self, day):
        """
        Pre-day execution hook.

        Args:
            day: The simulation day about to be executed.
        """
        self.day = day

    def after_day(self, tic=None, delete_previous=False):
        """
        Post-day execution hook.

        Automatically saves a checkpoint if the current day matches
        the checkpoint interval.

        Args:
            tic: Optional updated timer reference.
            delete_previous: Whether to delete the previous checkpoint file.
        """
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

    def on_error(self, error: Exception) -> Dict:
        """
        Hook called when a simulation error/crash occurs.

        Attempts to save an emergency checkpoint and constructs a
        failure summary dictionary.

        Args:
            error: The exception that triggered the crash.

        Returns:
            Dict: Error report containing policy, day, and exception info.
        """
        execution_time = time.process_time() - self.tic if self.tic else 0
        day = self.get_current_day()
        policy, sample_id = self.checkpoint.get_simulation_info().values()
        print(f"Crash in {policy} #{sample_id} at day {day}: {error}")

        traceback.print_exc()
        if self.checkpoint and self.state_getter:
            try:
                state_snapshot = self.state_getter()
                self.checkpoint.save_state(state_snapshot, self.day)
            except Exception as save_error:
                print(f"Failed to save emergency checkpoint: {save_error}")

        # Return error information instead of raising exception
        error_result = {
            "policy": policy,
            "sample_id": sample_id,
            "day": self.day,
            "error": str(error),
            "error_type": type(error).__name__,
            "execution_time": execution_time,
            "success": False,
        }
        return error_result

    def on_completion(self, policy=None, sample_id=None):
        """
        Hook called upon successful simulation completion.

        Clears temporary checkpoints and saves the final simulation state.

        Args:
            policy: Policy identifier for cleanup.
            sample_id: Sample identifier for cleanup.
        """
        if self.checkpoint:
            self.checkpoint.clear(policy, sample_id)
        state_snapshot = self.state_getter()
        self.checkpoint.save_state(state_snapshot, self.day, end_simulation=True)


class CheckpointError(Exception):
    """Special exception to carry error results through the context manager"""

    def __init__(self, error_result):
        """
        Initialize exception with error result dictionary.

        Args:
            error_result: Dictionary containing error details.
        """
        self.error_result = error_result
        super().__init__(error_result["error"])


@contextmanager
def checkpoint_manager(checkpoint, checkpoint_interval, state_getter, success_callback=None):
    """
    Context manager for automatic checkpoint lifecycle management.

    Wraps simulation loops with automatic checkpointing, error handling,
    and cleanup. Guarantees checkpoint cleanup on success and emergency
    saves on crashes.

    Args:
        checkpoint: SimulationCheckpoint instance
        checkpoint_interval: Days between automatic checkpoints
        state_getter: Callable[[], Any] returning current state
        success_callback: Optional callback on successful completion

    Yields:
        CheckpointHook: Hook object for manual before_day/after_day calls

    Raises:
        CheckpointError: Wraps original exception with error result dict

    Example:
        with checkpoint_manager(cp, 10, get_state) as hook:
            for day in range(100):
                hook.before_day(day)
                simulate_day(day)
                hook.after_day()
    """
    hook = CheckpointHook(checkpoint, checkpoint_interval, state_getter)
    try:
        yield hook
        hook.on_completion()
        if success_callback:
            success_callback()
    except Exception as e:
        error_result = hook.on_error(e)
        raise CheckpointError(error_result) from e
