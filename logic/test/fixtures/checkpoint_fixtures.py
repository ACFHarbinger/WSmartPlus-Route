"""
Fixtures for the Simulator pipeline - Checkpoint fixtures.
"""

import os
import pytest
from logic.src.pipeline.simulations.checkpoints.persistence import SimulationCheckpoint

@pytest.fixture
def basic_checkpoint(mocker, tmp_path):
    """
    Sets up a real (not mocked) SimulationCheckpoint in a temporary directory.
    - Mocks ROOT_DIR to a temporary path.
    """
    # Mock ROOT_DIR to point to a temporary path to avoid polluting real project
    mock_root = tmp_path / "mock_root"
    mock_root.mkdir(parents=True, exist_ok=True)
    mocker.patch("logic.src.pipeline.simulations.checkpoints.persistence.ROOT_DIR", str(mock_root))

    output_dir = tmp_path / "test_assets" / "results"
    cp = SimulationCheckpoint(
        output_dir=str(output_dir),
        checkpoint_dir="temp",
        policy="test_policy",
        sample_id=1,
    )

    os.makedirs(cp.checkpoint_dir, exist_ok=True)
    os.makedirs(cp.output_dir, exist_ok=True)

    return cp
