"""
Shared pytest fixtures and configuration for all test modules.

This file is automatically loaded by pytest and provides fixtures
that can be used across all test files.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# The project root is THREE levels up from conftest.py:
# conftest.py -> test -> logic -> WSmart-Route (Project Root)
project_root = Path(__file__).resolve().parent.parent.parent

# Add the project root to sys.path. This allows 'import logic.src...'
# to resolve 'logic' as a package within WSmart-Route/.
sys.path.insert(0, str(project_root))

# Register fixture plugins
pytest_plugins = [
    "logic.test.fixtures.arg_fixtures",
    "logic.test.fixtures.data_fixtures",
    "logic.test.fixtures.sim_fixtures",
    "logic.test.fixtures.policy_fixtures",
    "logic.test.fixtures.mrl_fixtures",
    "logic.test.fixtures.model_fixtures",
    "logic.test.fixtures.integration_fixtures",
    "logic.test.fixtures.eval_fixtures",
    "logic.test.fixtures.file_system_fixtures",
    "logic.test.fixtures.io_fixtures",
    "logic.test.fixtures.policy_aux_fixtures",
    "logic.test.fixtures.vectorized_policy_fixtures",
    "logic.test.fixtures.security_fixtures",
]


# ============================================================================
# Global / Shared Fixtures
# ============================================================================


@pytest.fixture
def mock_torch_device():
    """Returns a CPU torch device."""
    return torch.device("cpu")


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def temp_log_dir():
    """Create temporary log directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def temp_model_dir():
    """Create temporary model directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_sys_argv():
    """Fixture to mock sys.argv for argument parsing tests"""

    def _mock_argv(args):
        original_argv = sys.argv.copy()
        sys.argv = args
        yield
        sys.argv = original_argv

    return _mock_argv


@pytest.fixture
def mock_environment():
    """Fixture to mock environment variables"""

    def _mock_env(env_vars):
        original_env = os.environ.copy()
        os.environ.update(env_vars)
        yield
        os.environ.clear()
        os.environ.update(original_env)

    return _mock_env


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically setup test environment for all tests"""
    # Set test mode environment variable
    original_test_mode = os.environ.get("TEST_MODE")
    os.environ["TEST_MODE"] = "true"

    yield

    # Restore original environment
    if original_test_mode is None:
        os.environ.pop("TEST_MODE", None)
    else:
        os.environ["TEST_MODE"] = original_test_mode


@pytest.fixture
def disable_wandb():
    """Disable wandb logging for tests"""
    import os

    original_wandb_mode = os.environ.get("WANDB_MODE")
    os.environ["WANDB_MODE"] = "disabled"

    yield

    if original_wandb_mode is None:
        os.environ.pop("WANDB_MODE", None)
    else:
        os.environ["WANDB_MODE"] = original_wandb_mode


def _has_gurobi_license():
    """Returns True if Gurobi is installed and has a valid license."""
    try:
        import gurobipy as gp

        # Attempt to create a tiny model to verify license validity
        m = gp.Model("check")
        m.dispose()
        return True
    except Exception:
        return False


def _has_hexaly_license():
    """Returns True if Hexaly is installed and has a valid license."""
    try:
        import hexaly.optimizer

        # Attempt to create a context to verify license
        with hexaly.optimizer.HexalyOptimizer():
            pass
        return True
    except Exception:
        return False


@pytest.fixture(autouse=False)
def check_license(backend):
    """Check if the backend has a valid license."""
    if backend == "gurobi":
        if not _has_gurobi_license():
            pytest.skip("Gurobi license not found")
    elif backend == "hexaly":
        if not _has_hexaly_license():
            pytest.skip("Hexaly license not found")
    else:
        pytest.skip("Invalid backend")


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_root(request):
    """Cleanup test root directory after session."""

    def finalizer():
        """Finalizer to clean up test output."""
        path_to_clean = project_root / "assets" / "test_output"
        if path_to_clean.exists():
            shutil.rmtree(path_to_clean)

    request.addfinalizer(finalizer)


@pytest.fixture(scope="session", autouse=True)
def session_cleanup():
    """
    Cleanup artifacts that might be left over after tests,
    specifically 'test_dehb_output' directory and 'dummy.log'.
    """
    yield  # Run tests first

    artifacts_to_clean = [
        Path("test_dehb_output"),
        Path("dummy.log"),
        project_root / "assets" / "keys" / "testkey.pkl",
        project_root / "assets" / "keys" / "testkey.salt",
        project_root / "assets" / "output" / "2_days",
        project_root / "assets" / "output" / "5_days",
    ]

    # We use os.getcwd() to look in the current working directory where tests were run
    cwd = Path.cwd()

    for artifact_name in artifacts_to_clean:
        artifact_path = cwd / artifact_name
        if artifact_path.exists():
            try:
                if artifact_path.is_dir():
                    shutil.rmtree(artifact_path)
                else:
                    os.remove(artifact_path)
            except Exception:
                pass
