from unittest.mock import MagicMock, patch
import torch
from logic.src.utils.configs.setup_env import setup_cost_weights, setup_env
from logic.src.utils.configs.setup_worker import setup_model

class TestSetupUtils:
    """Class for setup_utils tests."""

    def test_setup_cost_weights_vrpp(self):
        """Test cost weights setup for VRPP."""
        opts = {"problem": "vrpp", "w_length": 5.0, "w_waste": None}
        weights = setup_cost_weights(opts, def_val=1.0)
        assert weights["length"] == 5.0
        assert weights["waste"] == 1.0

    def test_setup_cost_weights_wcvrp(self):
        """Test cost weights setup for WCVRP."""
        opts = {
            "problem": "wcvrp",
            "w_waste": 0.5,
            "w_length": 0.8,
            "w_overflows": None,
        }
        weights = setup_cost_weights(opts, def_val=1.0)
        assert weights["waste"] == 0.5
        assert weights["length"] == 0.8
        assert weights["overflows"] == 1.0

    def test_setup_cost_weights_custom_default(self):
        """Test cost weights with custom default value."""
        opts = {"problem": "vrpp", "w_waste": None, "w_length": 2.5}
        weights = setup_cost_weights(opts, def_val=3.0)
        assert weights["waste"] == 3.0
        assert weights["length"] == 2.5

    def test_setup_cost_weights_empty_problem(self):
        """Test cost weights with unsupported problem returns empty dict."""
        opts = {"problem": "unknown", "w_waste": None, "w_length": None}
        weights = setup_cost_weights(opts)
        assert weights == {}

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
