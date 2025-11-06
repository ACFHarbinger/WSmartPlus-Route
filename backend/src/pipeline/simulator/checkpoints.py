import os
import time
import pickle
import traceback

from backend.src.utils.definitions import ROOT_DIR
from datetime import datetime
from typing import Callable, Any, Dict
from contextlib import contextmanager


class SimulationCheckpoint:
    def __init__(self, output_dir, checkpoint_dir="temp", policy="", sample_id=0):
        self.checkpoint_dir = os.path.join(ROOT_DIR, checkpoint_dir)
        self.output_dir = os.path.join(output_dir, checkpoint_dir)
        self.policy = policy
        self.sample_id = sample_id

    def get_simulation_info(self):
        return {'policy': self.policy, 'sample': self.sample_id}
        
    def get_checkpoint_file(self, day=None, end_simulation=False):
        """Generate checkpoint filename for this policy/sample"""
        parent_dir = self.output_dir if end_simulation else self.checkpoint_dir
        if day is not None:
            return os.path.join(parent_dir, f"checkpoint_{self.policy}_{self.sample_id}_day{day}.pkl")
        last_day = self.find_last_checkpoint_day()
        return os.path.join(parent_dir, f"checkpoint_{self.policy}_{self.sample_id}_day{last_day}.pkl")
    
    def save_state(self, state, day=0, end_simulation=False):
        """Save simulation state with metadata"""
        checkpoint_data = {
            'state': state,
            'policy': self.policy,
            'sample_id': self.sample_id,
            'day': day,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        checkpoint_file = self.get_checkpoint_file(day, end_simulation)
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    
    def load_state(self, day=None):
        """Load simulation state - try specific day first, then latest"""
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
                    with open(checkpoint_file, 'rb') as f:
                        checkpoint_data = pickle.load(f)
                    
                    # Verify this checkpoint matches our policy/sample
                    if (checkpoint_data.get('policy') == self.policy and 
                        checkpoint_data.get('sample_id') == self.sample_id):
                        return checkpoint_data['state'], checkpoint_data.get('day', 0)
                    else:
                        print(f"Checkpoint mismatch: expected {self.policy}_{self.sample_id}")
                except Exception as e:
                    print(f"Error loading checkpoint {checkpoint_file}: {e}")
        print("Warning: no valid checkpoint found")
        return None, 0
    
    def find_last_checkpoint_day(self):
        """Find the highest day number with a checkpoint"""
        max_day = 0
        pattern = f"checkpoint_{self.policy}_{self.sample_id}_day"
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(pattern) and filename.endswith('.pkl'):
                try:
                    day_num = int(filename.split('day')[1].split('.pkl')[0])
                    max_day = max(max_day, day_num)
                except ValueError:
                    continue
        return max_day
    
    def clear(self, policy=None, sample_id=None):
        """Clear checkpoints for this policy/sample or all"""
        if policy is None:
            policy = self.policy
        if sample_id is None:
            sample_id = self.sample_id
            
        pattern = f"checkpoint_{policy}_{sample_id}"
        removed_count = 0
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(pattern) and filename.endswith('.pkl'):
                try:
                    os.remove(os.path.join(self.checkpoint_dir, filename))
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {filename}: {e}")
        return removed_count

    def delete_checkpoint_day(self, day):
        """Clear the checkpoint of this policy/sample for the specified day"""
        checkpoint_file = self.get_checkpoint_file(day)
        try:
            os.remove(os.path.join(self.checkpoint_dir, checkpoint_file))
            return True
        except Exception as e:
            print(f"Error removing {checkpoint_file}: {e}")
            return False


class CheckpointHook:
    def __init__(self, checkpoint, checkpoint_interval, state_getter=None):
        self.checkpoint = checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.day = 0
        self.tic = None
        self.state_getter = state_getter

    def get_current_day(self):
        return self.day
    
    def get_checkpoint_info(self):
        return {'checkpoint': self.checkpoint, 'interval': self.checkpoint_interval}
    
    def set_timer(self, tic):
        """Set the timer reference for execution time calculation"""
        self.tic = tic
    
    def set_state_getter(self, state_getter: Callable[[], Any]):
        """Set a function that returns the current state snapshot"""
        self.state_getter = state_getter
    
    def before_day(self, day):
        """Hook called before each day execution"""
        self.day = day
    
    def after_day(self, tic=None, delete_previous=False):
        """Hook called after each day execution - automatically saves checkpoint if needed"""
        previous_checkpoint_day = self.checkpoint.find_last_checkpoint_day()
        if tic: 
            self.tic = tic
        if (self.checkpoint and self.checkpoint_interval > 0 and 
            self.day % self.checkpoint_interval == 0 and self.state_getter):
            
            state_snapshot = self.state_getter()
            self.checkpoint.save_state(state_snapshot, self.day)
        if delete_previous: 
            self.checkpoint.delete_checkpoint_day(previous_checkpoint_day)

    def on_error(self, error: Exception) -> Dict:
        """
        Hook called when an error occurs
        Returns error result dictionary
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
            'policy': policy,
            'sample_id': sample_id,
            'day': self.day,
            'error': str(error),
            'error_type': type(error).__name__,
            'execution_time': execution_time,
            'success': False
        }
        return error_result
    
    def on_completion(self, policy=None, sample_id=None):
        """Hook called when simulation completes successfully"""
        if self.checkpoint:
            self.checkpoint.clear(policy, sample_id)
        state_snapshot = self.state_getter()
        self.checkpoint.save_state(state_snapshot, self.day, end_simulation=True)


class CheckpointError(Exception):
    """Special exception to carry error results through the context manager"""
    def __init__(self, error_result):
        self.error_result = error_result
        super().__init__(error_result['error'])


@contextmanager
def checkpoint_manager(checkpoint, checkpoint_interval, state_getter, success_callback=None):
    """
    Context manager that handles checkpoint logic automatically
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
