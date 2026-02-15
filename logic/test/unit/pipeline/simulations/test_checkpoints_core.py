import os
import pytest
from logic.src.pipeline.simulations.checkpoints.persistence import SimulationCheckpoint
from logic.src.pipeline.simulations.checkpoints.manager import CheckpointError, checkpoint_manager

class TestCheckpoints:
    """Class for SimulationCheckpoint tests."""

    @pytest.mark.unit
    def test_checkpoint_io(self, basic_checkpoint):
        """Test saving and loading a checkpoint."""
        state = {"data": "test_state"}
        day = 5
        basic_checkpoint.save_state(state, day)

        # Verify file exists with real pattern
        # checkpoint_{policy}_{sample_id}_day{day}.pkl
        pattern = f"checkpoint_{basic_checkpoint.policy}_{basic_checkpoint.sample_id}_day{day}.pkl"
        expected_file = os.path.join(basic_checkpoint.checkpoint_dir, pattern)
        assert os.path.exists(expected_file)

        # Load state
        loaded_state, loaded_day = basic_checkpoint.load_state(day=day)
        assert loaded_day == day
        assert loaded_state == state

    @pytest.mark.unit
    def test_checkpoint_resume_logic(self, basic_checkpoint):
        """Test finding the last checkpoint day."""
        # Create multiple checkpoints
        basic_checkpoint.save_state({"d": 1}, 10)
        basic_checkpoint.save_state({"d": 2}, 20)

        # find_last_checkpoint_day should return 20
        assert basic_checkpoint.find_last_checkpoint_day() == 20

        # Load without specifying day should get the latest (20)
        state, day = basic_checkpoint.load_state()
        assert day == 20
        assert state == {"d": 2}

    @pytest.mark.unit
    def test_checkpoint_error_handling(self, basic_checkpoint, mocker):
        """Test error handling during checkpoint operations using manager."""
        def failing_task():
            raise ValueError("Test error")

        with pytest.raises(CheckpointError) as excinfo:
            with checkpoint_manager(basic_checkpoint, 1, lambda: {"state": 1}):
                failing_task()

        assert "Test error" in str(excinfo.value)
        # CheckpointError stores the error_result dict
        assert excinfo.value.error_result["error"] == "Test error"

    @pytest.mark.unit
    def test_get_checkpoint_file_explicit(self, basic_checkpoint):
        """Test get_checkpoint_file with explicit day."""
        path = basic_checkpoint.get_checkpoint_file(day=10)
        pattern = f"checkpoint_{basic_checkpoint.policy}_{basic_checkpoint.sample_id}_day10.pkl"
        assert pattern in path

    @pytest.mark.unit
    def test_load_state_file_not_found(self, basic_checkpoint):
        """Test load_state when file is missing."""
        # Ensure dir is empty for this test
        basic_checkpoint.clear()
        state, day = basic_checkpoint.load_state(day=99)
        assert state is None
        assert day == 0
