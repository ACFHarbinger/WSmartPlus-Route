"""Tests for setup_utils.py."""

from unittest.mock import MagicMock

import torch
from logic.src.utils.configs.setup_env import setup_cost_weights
from logic.src.utils.configs.setup_optimization import setup_optimizer_and_lr_scheduler


def test_setup_cost_weights():
    """Verify cost weight setup."""
    opts = {
        "problem": "vrpp",
        "w_waste": None,
        "w_length": 2.0,
    }
    cw = setup_cost_weights(opts, def_val=1.0)
    assert cw["waste"] == 1.0
    assert cw["length"] == 2.0
    assert opts["w_waste"] == 1.0


def test_setup_optimizer_and_lr_scheduler():
    """Verify optimizer and scheduler setup."""
    model = torch.nn.Linear(10, 1)
    baseline = MagicMock()
    baseline.get_learnable_parameters.return_value = []

    opts = {"optimizer": "adam", "lr_model": 1e-3, "lr_scheduler": "exp", "lr_decay": 0.99, "device": "cpu"}

    opt, sched = setup_optimizer_and_lr_scheduler(model, baseline, {}, opts)
    assert isinstance(opt, torch.optim.Adam)
    assert isinstance(sched, torch.optim.lr_scheduler.ExponentialLR)


def test_setup_optimizer_available_types():
    """Verify various optimizer types."""
    model = torch.nn.Linear(1, 1)
    baseline = MagicMock()
    baseline.get_learnable_parameters.return_value = []

    for opt_name in ["adamax", "adamw", "rmsprop", "sgd"]:
        opts = {
            "optimizer": opt_name,
            "lr_model": 1e-3,
            "lr_scheduler": "step",
            "lrs_step_size": 1,
            "lr_decay": 0.9,
            "device": "cpu",
        }
        opt, _ = setup_optimizer_and_lr_scheduler(model, baseline, {}, opts)
        assert opt.__class__.__name__.lower().startswith(opt_name)


from logic.src.utils.configs.setup_utils import setup_env, setup_model, setup_hrl_manager, setup_model_and_baseline
import os
from unittest.mock import patch


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

    with patch("logic.src.utils.configs.setup_utils.load_model", return_value=(mock_model, mock_configs)):
        model, configs = setup_model(
            "am_50", "path/", {"am": "am.pt"}, torch.device("cpu"), lock
        )
        assert model == mock_model
        assert configs == mock_configs
        mock_model.set_decode_type.assert_called()


def test_setup_hrl_manager_not_hrl():
    """Returns None if method is not hrl."""
    opts = {"mrl_method": "reinforce"}
    assert setup_hrl_manager(opts, torch.device("cpu"), configs={"mrl_method": "none"}) is None


def test_setup_model_and_baseline_basic():
    """Test full model and baseline assembly."""
    opts = {
        "encoder": "gat",
        "model": "am",
        "embed_dim": 16,
        "hidden_dim": 32,
        "n_encode_layers": 1,
        "n_encode_sublayers": 1,
        "n_decode_layers": 1,
        "n_heads": 2,
        "normalization": "batch",
        "learn_affine": True,
        "track_stats": False,
        "epsilon_alpha": 1e-5,
        "momentum_beta": 0.1,
        "lrnorm_k": 1.0,
        "gnorm_groups": 2,
        "activation": "relu",
        "af_param": 1.0,
        "af_threshold": 6.0,
        "af_replacement": 6.0,
        "af_nparams": 1,
        "af_urange": [0.1, 0.2],
        "dropout": 0.1,
        "aggregation": "mean",
        "aggregation_graph": "mean",
        "tanh_clipping": 10.0,
        "mask_inner": True,
        "mask_logits": True,
        "mask_graph": False,
        "checkpoint_encoder": False,
        "shrink_size": 1,
        "temporal_horizon": 1,
        "n_predict_layers": 1,
        "device": "cpu",
        "baseline": "exponential",
        "exp_beta": 0.8,
        "bl_warmup_epochs": 0
    }

    mock_problem = MagicMock()

    with patch("logic.src.models.AttentionModel") as mock_am, \
         patch("logic.src.models.ExponentialBaseline") as mock_bl:

        setup_model_and_baseline(mock_problem, {}, False, opts)
        assert mock_am.called
        assert mock_bl.called
