from unittest.mock import MagicMock, patch
import torch
from logic.src.utils.configs.setup_env import setup_env
from logic.src.utils.configs.setup_worker import setup_model


class TestSetupUtils:
    """Class for setup_utils tests."""

    @patch("logic.src.utils.configs.setup_worker.load_model")
    def test_setup_model_am(self, mock_load):
        """Test setup_model for Attention Model."""
        mock_model = MagicMock()
        mock_configs = {"embed_dim": 128}
        mock_load.return_value = (mock_model, mock_configs)
        device = torch.device("cpu")
        lock = MagicMock()
        model, configs = setup_model("am_greedy", "path", {"am": "model.pt"}, device, lock)
        assert model == mock_model
        assert configs == mock_configs
        mock_load.assert_called_once()

    @patch("gurobipy.Env", create=True)
    def test_setup_env_gurobi(self, mock_gp_env):
        """Test setup_env for Gurobi."""
        mock_env = MagicMock()
        mock_gp_env.return_value = mock_env
        env = setup_env("gurobi")
        assert env == mock_env or env is None
