"""Tests for setup_utils.py."""

from unittest.mock import MagicMock, patch

import torch

from logic.src.utils.configs.setup_env import setup_env
from logic.src.utils.configs.setup_worker import setup_model
from logic.src.utils.configs.setup_manager import setup_hrl_manager
import os


def test_setup_env_non_vrpp():
    """Returns None for non-vrpp problems."""
    assert setup_env("tsp") is None


def test_setup_env_gurobi_local(tmp_path):
    """Test local Gurobi setup with license file."""
    lic_file = tmp_path / "gurobi.lic"
    lic_file.write_text("dummy license")

    with patch("logic.src.constants.ROOT_DIR", str(tmp_path)), \
         patch("os.path.exists", return_value=True), \
         patch("gurobipy.Env") as mock_env:
        setup_env("vrpp", server=False, gplic_filename="gurobi.lic")
        assert os.environ["GRB_LICENSE_FILE"].endswith("gurobi.lic")


def test_setup_model(tmp_path):
    """Test model loading utility."""
    mock_model = MagicMock()
    mock_configs = {"key": "val"}
    lock = MagicMock()

    with patch("logic.src.utils.functions.load_model", return_value=(mock_model, mock_configs)):
        model, configs = setup_model(
            "am_50", "path/", {"am": "am.pt"}, torch.device("cpu"), lock
        )
        assert model == mock_model
        assert configs == mock_configs
        mock_model.set_strategy.assert_called()


def test_setup_hrl_manager_not_hrl():
    """Returns None if method is not hrl."""
    opts = {"mrl_method": "reinforce"}
    assert setup_hrl_manager(opts, torch.device("cpu"), configs={"mrl_method": "none"}) is None
