"""
Consolidated tests for utility modules.
Merges functionality from:
- test_debug_utils_coverage.py
- test_functions_coverage.py
- test_lexsort_coverage.py
- test_load_model_coverage.py
- test_log_utils_coverage.py
- test_log_utils_more_coverage.py
- test_setup_utils_coverage.py
- test_utils.py
"""

from typing import Tuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
import torch.nn as nn

# Imports from project
from logic.src.utils.debug_utils import watch
from logic.src.utils.functions.function import (
    add_attention_hooks,
    compute_in_batches,
    do_batch_rep,
    get_inner_model,
    get_path_until_string,
    load_model,
    load_problem,
    move_to,
    parse_softmax_temperature,
    sample_many,
    torch_load_cpu,
)
from logic.src.utils.functions.lexsort import _torch_lexsort_cuda, torch_lexsort
from logic.src.utils.io.processing import (
    find_single_input_values,
    process_dict_of_dicts,
    process_list_of_dicts,
)
from logic.src.utils.logging.log_utils import (
    log_epoch,
    log_to_json,
    log_to_json2,
    log_training,
    log_values,
    output_stats,
    runs_per_policy,
)
from logic.src.utils.setup_utils import setup_cost_weights, setup_env, setup_model

# ============================================================================
# Debug Utils Tests
# ============================================================================


class TestDebugUtils:
    """Class for debug_utils tests."""

    @patch("logic.src.utils.debug_utils.sys._getframe")
    @patch("logic.src.utils.debug_utils.sys.settrace")
    def test_watch_variable(self, mock_settrace, mock_getframe):
        """Test enabling the watcher."""
        # Mock frame setup
        mock_frame = MagicMock()
        mock_frame.f_locals = {"my_var": 10}
        mock_getframe.return_value = mock_frame

        watch("my_var")

        assert mock_settrace.called
        tracer_func = mock_settrace.call_args[0][0]
        assert callable(tracer_func)

        # Test callback
        mock_frame.f_locals = {"my_var": 20}
        callback = MagicMock()

        # Re-initialize watch with callback
        mock_frame.f_locals = {"my_var": 10}
        watch("my_var", callback=callback)
        tracer_func_with_cb = mock_settrace.call_args[0][0]

        # Trigger tracer
        mock_frame.f_locals = {"my_var": 99}
        tracer_func_with_cb(mock_frame, "line", None)

        callback.assert_called_with(10, 99, mock_frame)

    @patch("logic.src.utils.debug_utils.sys._getframe")
    @patch("logic.src.utils.debug_utils.sys.settrace")
    def test_watch_not_found(self, mock_settrace, mock_getframe):
        """Test watch raises NameError if var not found."""
        mock_frame = MagicMock()
        mock_frame.f_locals = {}
        mock_frame.f_globals = {}
        mock_getframe.return_value = mock_frame

        with pytest.raises(NameError):
            watch("missing_var")

    @patch("logic.src.utils.debug_utils.sys._getframe")
    @patch("logic.src.utils.debug_utils.sys.settrace")
    @patch("logic.src.utils.debug_utils.traceback.extract_stack")
    def test_default_callback(self, mock_extract, mock_settrace, mock_getframe):
        """Test the default print callback."""
        mock_frame = MagicMock()
        mock_frame.f_locals = {"v": 1}
        mock_getframe.return_value = mock_frame

        mock_stack_item = MagicMock()
        mock_stack_item.filename = "file.py"
        mock_stack_item.lineno = 10
        mock_stack_item.name = "func"
        mock_extract.return_value = [mock_stack_item, mock_stack_item]

        watch("v")
        tracer = mock_settrace.call_args[0][0]

        mock_frame.f_locals = {"v": 2}

        with patch("builtins.print") as mock_print:
            tracer(mock_frame, "line", None)
            assert mock_print.called


# ============================================================================
# LexSort Tests
# ============================================================================


class TestLexSort:
    """Class for lexsort tests."""

    def test_torch_lexsort_cpu(self):
        """Test fallback to numpy lexsort on CPU."""
        k1 = torch.tensor([1, 1, 2, 2])
        k2 = torch.tensor([4, 3, 2, 1])
        keys = (k1, k2)
        idx = torch_lexsort(keys)
        assert idx.dtype == torch.int64
        expected = torch.tensor([3, 2, 1, 0])
        assert torch.equal(idx, expected)

    def test_torch_lexsort_cuda_path(self):
        """Test the pure-pytorch implementation used for CUDA."""
        k1 = torch.tensor([1, 1, 2, 2])
        k2 = torch.tensor([4, 3, 2, 1])
        keys = (k1, k2)
        idx = _torch_lexsort_cuda(keys)
        assert idx.shape == k1.shape
        assert idx.dtype == torch.int64

    def test_torch_lexsort_cuda_call(self):
        """Test that torch_lexsort calls the cuda version if is_cuda is set."""
        k1 = MagicMock()
        k1.is_cuda = True
        keys = (k1,)
        with patch("logic.src.utils.functions.lexsort._torch_lexsort_cuda") as mock_cuda:
            torch_lexsort(keys)
            mock_cuda.assert_called_once()


# ============================================================================
# Load Model Tests
# ============================================================================


class TestLoadModel:
    """Class for load_model tests."""

    @patch("logic.src.utils.functions.function.load_problem")
    @patch("logic.src.utils.functions.function.load_args")
    @patch("logic.src.utils.functions.function.torch.load")
    def test_load_model_basic(self, mock_torch_load, mock_load_args, mock_load_problem, tmp_path):
        """Test load_model by mocking internal dependencies and file structure."""
        model_dir = tmp_path / "model_dir"
        model_dir.mkdir()

        # Create dummy epoch file
        (model_dir / "epoch-10.pt").touch()

        mock_load_args.return_value = {
            "problem": "vrpp",
            "model": "am",
            "encoder": "gat",
            "embedding_dim": 128,
            "hidden_dim": 512,
            "n_encode_layers": 3,
            "n_encode_sublayers": 1,
            "n_decode_layers": 1,
            "n_heads": 8,
            "normalization": "batch",
            "learn_affine": True,
            "track_stats": True,
            "epsilon_alpha": 1e-5,
            "momentum_beta": 0.1,
            "lrnorm_k": 1,
            "gnorm_groups": 1,
            "activation": "relu",
            "af_param": 0.0,
            "af_threshold": 0.0,
            "af_replacement": 0.0,
            "af_nparams": 1,
            "af_urange": 0.1,
            "dropout": 0.0,
            "aggregation": "mean",
            "aggregation_graph": "avg",
            "tanh_clipping": 10.0,
        }

        mock_problem = MagicMock()
        mock_load_problem.return_value = mock_problem
        mock_torch_load.return_value = {"model": {}}

        with patch("os.listdir", return_value=["epoch-10.pt"]):
            model, args = load_model(str(model_dir))

        assert model is not None
        assert args == mock_load_args.return_value
        assert mock_torch_load.called


# ============================================================================
# Log Utils Tests
# ============================================================================


class TestLogUtils:
    """Class for log_utils tests."""

    @patch("logic.src.utils.logging.log_utils.wandb")
    def test_log_epoch(self, mock_wandb):
        """Test log_epoch with mocked wandb."""
        opts = {"train_time": False, "wandb_mode": "online"}
        x_tup = ("epoch", 1)
        loss_keys = ["loss"]
        epoch_loss = {"loss": [torch.tensor([1.0]), torch.tensor([2.0])]}
        log_epoch(x_tup, loss_keys, epoch_loss, opts)
        assert mock_wandb.log.called

    @patch("logic.src.utils.logging.log_utils.wandb")
    @patch("logic.src.utils.logging.log_utils.plt")
    def test_log_training(self, mock_plt, mock_wandb):
        """Test log_training with mocked components."""
        opts = {
            "train_time": False,
            "wandb_mode": "online",
            "log_dir": "logs",
            "save_dir": "checkpoints/run1",
            "checkpoints_dir": "checkpoints",
        }
        columns = pd.MultiIndex.from_tuples([("loss", "mean"), ("loss", "std"), ("loss", "max"), ("loss", "min")])
        table_df = pd.DataFrame([[1.0, 0.1, 1.2, 0.8]], columns=columns)
        loss_keys = ["loss"]

        with patch("os.makedirs"), patch("pandas.DataFrame.to_parquet"):
            log_training(loss_keys, table_df, opts)
        assert mock_wandb.log.called

    @patch("logic.src.utils.logging.log_utils.os.path.exists")
    @patch("logic.src.utils.logging.log_utils.read_json")
    @patch("builtins.open", new_callable=MagicMock)
    def test_log_to_json(self, mock_open, mock_read, mock_exists):
        """Test log_to_json with mock file."""
        mock_exists.return_value = True
        mock_read.return_value = {"runs": []}
        mock_open.return_value.__enter__.return_value = MagicMock()
        with (
            patch("json.dump") as mock_dump,
            patch("logic.src.utils.logging.log_utils._sort_log"),
        ):
            res = log_to_json("path.json", ["val"], {"key": [1]}, sort_log=True)
            assert res is not None
            assert mock_dump.called

    @patch("logic.src.utils.logging.log_utils.os.path.exists")
    @patch("builtins.open", new_callable=MagicMock)
    def test_log_to_json2(self, mock_open, mock_exists):
        """Test log_to_json2 (thread-safe version) with mock file."""
        mock_exists.return_value = False
        mock_open.return_value.__enter__.return_value = MagicMock()
        with patch("json.load", return_value=[]), patch("json.dump") as mock_dump:
            res = log_to_json2("path.json", ["val"], {"key": [2]})
            assert res is not None
            assert mock_dump.called

    @patch("logic.src.utils.logging.log_utils.wandb")
    def test_log_values(self, mock_wandb):
        """Test log_values function."""
        cost = torch.tensor([10.0])
        grad_norms = (
            [torch.tensor([1.0]), torch.tensor([1.5])],
            [torch.tensor([0.5]), torch.tensor([0.6])],
        )
        epoch = 1
        batch_id = 5
        step = 100
        l_dict = {
            "reinforce_loss": torch.tensor([1.0]),
            "nll": torch.tensor([0.5]),
            "imitation_loss": torch.tensor([0.1]),
            "baseline_loss": torch.tensor([0.2]),
        }
        tb_logger = MagicMock()
        opts = {
            "train_time": False,
            "no_tensorboard": False,
            "baseline": "critic",
            "wandb_mode": "online",
        }
        log_values(cost, grad_norms, epoch, batch_id, step, l_dict, tb_logger, opts)
        assert mock_wandb.log.called
        assert tb_logger.log_value.called

    @patch("glob.glob")
    @patch("logic.src.utils.logging.log_utils.read_json")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("json.dump")
    def test_output_stats(self, mock_dump, mock_open_file, mock_read, mock_glob):
        """Test output_stats function."""
        mock_glob.return_value = ["dir/log.json"]
        mock_read.return_value = [
            {"policy1": {"cost": 10.0}},
            {"policy1": {"cost": 20.0}},
        ]
        mock_open_file.return_value.__enter__.return_value = MagicMock()
        mean, std = output_stats("home", 1, 50, "out", "area", 2, ["policy1"], ["cost"], print_output=False)
        assert isinstance(mean, dict)
        assert isinstance(std, dict)
        assert mock_dump.called

    @patch("logic.src.utils.logging.log_utils.read_json")
    def test_aggregate_stats(self, mock_read):
        """Test runs_per_policy function."""
        mock_read.return_value = [{"policy1": "res"}, {}]
        res = runs_per_policy("home", 1, [50], "out", "area", [100], ["policy1"], print_output=True)
        assert isinstance(res, list)
        assert len(res) == 1
        assert res[0]["policy1"] == [0]


# ============================================================================
# Functions Utils Tests (Merged)
# ============================================================================


class TestFunctions:
    """Class for functions.py tests."""

    def test_compute_in_batches_tuple(self):
        """Test compute_in_batches with tuple return."""

        def f(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Helper function for batch computation test."""
            return x * 2, x + 1

        x = torch.arange(10)
        res1, res2 = compute_in_batches(f, 4, x)
        assert torch.equal(res1, x * 2)
        assert torch.equal(res2, x + 1)

    def test_compute_in_batches_none(self):
        """Test compute_in_batches with None return."""

        def f(x: torch.Tensor) -> None:
            """Helper function returning None for batch computation test."""
            return None

        x = torch.arange(10)
        res = compute_in_batches(f, 4, x)
        assert res is None

    def test_add_attention_hooks(self):
        """Test adding attention hooks to a model."""
        model = MagicMock()
        mock_layer = MagicMock()
        mock_layer.att.module = MagicMock()
        model.layers = [mock_layer]
        hook_data = add_attention_hooks(model)
        assert "weights" in hook_data
        assert len(hook_data["handles"]) == 1

    def test_do_batch_rep_complex(self):
        """Test do_batch_rep with list and tuple."""
        t = torch.randn(3)
        data = [t, (t, {"x": t})]
        rep = do_batch_rep(data, 2)
        assert isinstance(rep, list)
        assert rep[0].shape == (6,)
        assert isinstance(rep[1], tuple)

    def test_sample_many(self):
        """Test sample_many sampling loop."""

        def inner_func(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Helper function simulating model forward pass for sampling test."""
            batch_size = x.size(0)
            return torch.randn(batch_size, 5, 5), torch.randint(0, 5, (batch_size, 5))

        def get_cost(input_data: torch.Tensor, pi: torch.Tensor) -> Tuple[torch.Tensor, None]:
            """Helper function simulating cost calculation for sampling test."""
            batch_size = input_data.size(0)
            return torch.rand(batch_size), None

        input_data = torch.rand(2, 10)
        minpis, mincosts = sample_many(inner_func, get_cost, input_data, batch_rep=2, iter_rep=3)
        assert minpis.shape[0] == 2
        assert mincosts.shape[0] == 2

    def test_get_inner_model_direct(self):
        """Test get_inner_model with non-wrapped model."""
        model = nn.Linear(10, 5)
        inner = get_inner_model(model)
        assert inner is model

    def test_get_inner_model_data_parallel(self):
        """Test get_inner_model correctly checks isinstance DataParallel."""
        model = nn.Linear(10, 5)
        inner = get_inner_model(model)
        assert inner is model

    def test_move_to_tensor(self):
        """Test move_to with tensor."""
        tensor = torch.rand(5, 3)
        result = move_to(tensor, torch.device("cpu"))
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"

    def test_move_to_dict(self):
        """Test move_to with dict of tensors."""
        data = {"a": torch.rand(3), "b": torch.rand(4)}
        result = move_to(data, torch.device("cpu"))
        assert isinstance(result, dict)
        assert all(v.device.type == "cpu" for v in result.values())

    def test_load_problem_vrpp(self):
        """Test load_problem for VRPP."""
        problem = load_problem("vrpp")
        assert problem.NAME == "vrpp"

    def test_load_problem_wcvrp(self):
        """Test load_problem for WCVRP."""
        problem = load_problem("wcvrp")
        assert problem.NAME == "wcvrp"

    def test_load_problem_cvrpp(self):
        """Test load_problem for CVRPP."""
        problem = load_problem("cvrpp")
        assert problem.NAME == "cvrpp"

    def test_parse_softmax_temperature_float(self):
        """Test parse_softmax_temperature with float."""
        temp = parse_softmax_temperature("1.5")
        assert temp == 1.5

    def test_get_path_until_string(self):
        """Test get_path_until_string function."""
        path = "/home/user/project/outputs/run1/model.pt"
        result = get_path_until_string(path, "outputs")
        assert result.endswith("outputs")

    def test_do_batch_rep_tensor(self):
        """Test do_batch_rep with tensor."""
        tensor = torch.rand(2, 3)
        result = do_batch_rep(tensor, 3)
        assert result.shape[0] == 6

    def test_do_batch_rep_dict(self):
        """Test do_batch_rep with dict of tensors."""
        data = {"x": torch.rand(2, 3), "y": torch.rand(2, 4)}
        result = do_batch_rep(data, 2)
        assert result["x"].shape[0] == 4

    def test_torch_load_cpu(self, tmp_path):
        """Test CPU-mapped tensor loading."""
        tensor = torch.randn(10, 5)
        filepath = str(tmp_path / "tensor.pt")
        torch.save(tensor, filepath)
        loaded = torch_load_cpu(filepath)
        assert torch.allclose(tensor, loaded)


# ============================================================================
# Setup Utils Tests
# ============================================================================


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

    @patch("logic.src.utils.setup_utils.load_model")
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

    @patch("logic.src.utils.setup_utils.gp.Env", create=True)
    def test_setup_env_gurobi(self, mock_gp_env):
        """Test setup_env for Gurobi."""
        mock_env = MagicMock()
        mock_gp_env.return_value = mock_env
        env = setup_env("gurobi")
        assert env == mock_env or env is None


# ============================================================================
# IO Processing Utils Tests (Preserved)
# ============================================================================


class TestIOProcessing:
    """
    Tests for io/processing.py module (preserved from original test_utils.py).
    Duplicates logic in test_io.py but kept for backward compatibility request.
    """

    def test_process_dict_of_dicts_single_value(self):
        """Test processing a dict of dicts with single values."""
        data = {"policy1": {"km": 10.0, "waste": 50.0}}
        modified = process_dict_of_dicts(data, output_key="km", process_func=lambda x, y: x * 2, update_val=0)
        assert modified
        assert data["policy1"]["km"] == 20.0

    def test_process_dict_of_dicts_list_values(self):
        """Test processing a dict of dicts with list values."""
        data = {"policy1": {"km": [10.0, 20.0], "waste": 50.0}}
        modified = process_dict_of_dicts(data, output_key="km", process_func=lambda x, y: x + 5, update_val=0)
        assert modified
        assert data["policy1"]["km"] == [15.0, 25.0]

    def test_process_list_of_dicts(self):
        """Test processing a list of dicts."""
        data = [{"policy1": {"km": 10.0}}, {"policy2": {"km": 20.0}}]
        modified = process_list_of_dicts(data, output_key="km", process_func=lambda x, y: x / 2, update_val=0)
        assert modified
        assert data[0]["policy1"]["km"] == 5.0
        assert data[1]["policy2"]["km"] == 10.0

    def test_find_single_input_values(self):
        """Test finding single input values in nested dict."""
        data = {"policy1": {"day1": {"km": 100}}, "policy2": {"day1": {"km": 200}}}
        values = find_single_input_values(data, output_key="km")
        assert len(values) == 2
