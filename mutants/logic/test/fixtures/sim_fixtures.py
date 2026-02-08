"""
Fixtures for the Simulator pipeline.
"""

import os
from multiprocessing import Lock, Value
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from logic.src.pipeline.simulations.bins import Bins
from logic.src.pipeline.simulations.day_context import SimulationDayContext


@pytest.fixture
def wsr_opts(tmp_path):
    """
    Provides a dictionary of default options (opts) for simulation tests.
    Tests can modify this fixture's output as needed.
    """
    # Create necessary subdirectories in tmp_path
    (tmp_path / "data" / "wsr_simulator").mkdir(parents=True, exist_ok=True)
    results_dir = tmp_path / "assets" / "test_output" / "10_days" / "test_area_2"
    results_dir.mkdir(parents=True, exist_ok=True)

    return {
        "days": 10,
        "size": 2,
        "area": "riomaior",
        "waste_type": "paper",
        "policies": ["test_policy_gamma1"],
        "output_dir": "test_output",
        "checkpoint_dir": "temp",
        "resume": False,
        "distance_method": "hsd",
        "dm_filepath": None,
        "env_file": None,
        "gapik_file": None,
        "symkey_name": None,
        "edge_threshold": 50,
        "edge_method": "knn",
        "temperature": 1.0,
        "decode_type": "greedy",
        "vertex_method": "mmn",
        "server_run": False,
        "gplic_file": None,
        "cache_regular": False,
        "waste_filepath": None,
        "run_tsp": False,
        "n_vehicles": 1,
        "checkpoint_days": 5,
        "no_progress_bar": True,
        "data_distribution": "gamma",
        "n_samples": 1,
        "seed": 42,
        "problem": "vrpp",
        "stats_filepath": None,
        "model_path": None,
    }


@pytest.fixture
def basic_bins(tmp_path):
    """Returns a basic Bins instance for testing."""
    return Bins(
        n=10,
        data_dir=str(tmp_path),
        sample_dist="gamma",
        area="riomaior",
        waste_type="paper",
    )


@pytest.fixture
def basic_checkpoint(mocker, tmp_path):
    """
    Sets up a mocked environment for SimulationCheckpoint testing.
    - Mocks ROOT_DIR to a temporary path.
    - Mocks os.listdir to prevent FileNotFoundError.
    """

    # 1. Mock ROOT_DIR to point to a temporary path
    mock_root = tmp_path / "mock_root"
    mocker.patch("logic.src.pipeline.simulations.checkpoints.persistence.ROOT_DIR", new=str(mock_root))

    # 2. Define the path structure passed to the constructor
    # The test implies the output_dir input ends with "results"
    mock_output_dir_base = tmp_path / "test_assets" / "results"

    # 3. Mock os.listdir since it's used by find_last_checkpoint_day() (called by get_checkpoint_file without 'day')
    mocker.patch("os.listdir", return_value=[])

    # 4. Initialize SimulationCheckpoint
    from logic.src.pipeline.simulations.checkpoints import SimulationCheckpoint

    cp = SimulationCheckpoint(
        # Pass the path that ends in 'results'
        output_dir=str(mock_output_dir_base),
        checkpoint_dir="temp",
        policy="test_policy",
        sample_id=1,
    )

    # Ensure the checkpoint directories exist for later tests (like save/load)
    os.makedirs(cp.checkpoint_dir, exist_ok=True)
    os.makedirs(cp.output_dir, exist_ok=True)

    return cp


@pytest.fixture
def mock_lock_counter(mocker):
    """
    Provides mock multiprocessing.Lock and Value objects.
    """
    mock_lock = mocker.MagicMock(spec=Lock())
    mock_counter = mocker.MagicMock(spec=Value("i"))
    mock_counter.value = 0
    # Mock the context manager for the lock
    mock_counter_lock = mocker.MagicMock()
    mock_counter_lock.__enter__.return_value = None
    mock_counter_lock.__exit__.return_value = None
    return mock_lock, mock_counter


@pytest.fixture
def mock_bins_instance(mocker):
    """Returns a high-level mock of a Bins instance."""
    mock_bins = mocker.MagicMock()
    mock_bins.inoverflow = np.array([10, 20])
    mock_bins.collected = np.array([100, 200])
    mock_bins.ncollections = np.array([1, 2])
    mock_bins.lost = np.array([5, 5])
    mock_bins.travel = 50.0
    mock_bins.ndays = 10
    mock_bins.n = 2
    mock_bins.profit = 100.0
    mock_bins.get_fill_history.return_value = np.array([[10, 20], [30, 40]])
    mock_bins.stochasticFilling.return_value = (
        0,
        np.zeros(2),
        np.zeros(2),
        0,
    )
    mock_bins.c = np.full(2, 50.0)  # 50% fill
    mock_bins.means = np.full(2, 10.0)
    mock_bins.std = np.full(2, 1.0)
    mock_bins.collectlevl = np.full(2, 90.0)  # For last_minute policy
    mock_bins.collect.return_value = (100.0, 2, 10.0, 5.0)

    return mock_bins


@pytest.fixture
def mock_sim_dependencies(mocker, tmp_path, mock_bins_instance):
    """
    Mocks all major external dependencies for single_simulation
    and sequential_simulations.
    """
    # 1. Patch ROOT_DIR in both modules to ensure consistency
    mocker.patch("logic.src.pipeline.simulations.states.initializing.ROOT_DIR", str(tmp_path))
    mocker.patch("logic.src.pipeline.simulations.states.base.context.ROOT_DIR", str(tmp_path))
    mocker.patch("logic.src.pipeline.simulations.simulator.ROOT_DIR", str(tmp_path))

    # 2. Mock loader functions
    mock_depot = pd.DataFrame({"ID": [0], "Lat": [40], "Lng": [-8], "Stock": [0], "Accum_Rate": [0]})
    mock_data = pd.DataFrame({"ID": [1, 2], "Stock": [10, 20], "Accum_Rate": [0.1, 0.2]})
    mock_coords = pd.DataFrame({"ID": [1, 2], "Lat": [40.1, 40.2], "Lng": [-8.1, -8.2]})
    mocker.patch("logic.src.pipeline.simulations.processor.load_depot", return_value=mock_depot)
    mocker.patch(
        "logic.src.pipeline.simulations.processor.load_simulator_data",
        return_value=(mock_data.copy(), mock_coords.copy()),
    )

    # Mock setup_basedata in states to return the tuple directly
    mock_setup_basedata = mocker.patch(
        "logic.src.pipeline.simulations.states.initializing.setup_basedata",
        return_value=(mock_data.copy(), mock_coords.copy(), mock_depot.copy()),
    )

    # 3. Mock processor functions (patched in states where imported)
    mock_proc_data = pd.DataFrame({"ID": [0, 1, 2], "Stock": [0, 10, 20]})
    mock_proc_coords = pd.DataFrame({"ID": [0, 1, 2], "Lat": [40, 40.1, 40.2], "Lng": [-8, -8.1, -8.2]})
    mock_process_data = mocker.patch(
        "logic.src.pipeline.simulations.states.initializing.process_data",
        return_value=(mock_proc_data.copy(), mock_proc_coords.copy()),
    )
    mocker.patch(
        "logic.src.pipeline.simulations.states.initializing.process_model_data",
        return_value=("mock_model_tup_0", "mock_model_tup_1"),
    )

    # 4. Mock network functions (patched in states where imported)
    mock_dist_tup = (
        np.array([[0, 1], [1, 0]]),
        "mock_paths",
        "mock_tensor",
        "mock_distC",
    )
    mock_adj_matrix = np.array([[1, 1], [1, 1]])
    mock_setup_dist = mocker.patch(
        "logic.src.pipeline.simulations.states.initializing.setup_dist_path_tup",
        return_value=(mock_dist_tup, mock_adj_matrix),
    )
    mocker.patch(
        "logic.src.pipeline.simulations.processor.compute_distance_matrix",
        return_value=np.array([[0, 1], [1, 0]]),
    )
    mocker.patch(
        "logic.src.pipeline.simulations.processor.apply_edges",
        return_value=("mock_dist_edges", "mock_paths", "mock_adj"),
    )
    mocker.patch(
        "logic.src.pipeline.simulations.processor.get_paths_between_states",
        return_value="mock_all_paths",
    )

    # 5. Mock Bins class
    mocker.patch("logic.src.pipeline.simulations.bins.Bins", return_value=mock_bins_instance)

    # 6. Mock setup functions
    mock_setup_model = mocker.patch(
        "logic.src.pipeline.simulations.states.initializing.setup_model",
        return_value=(MagicMock(), MagicMock()),
    )
    mock_setup_env = mocker.patch("logic.src.pipeline.simulations.states.initializing.setup_env", return_value="mock_or_env")

    # 7. Mock day function
    mock_dlog = {
        "day": 1,
        "overflows": 0,
        "kg_lost": 0,
        "kg": 0,
        "ncol": 0,
        "km": 0,
        "kg/km": 0,
        "tour": [0],
    }

    mock_return_ctx = mocker.MagicMock()
    mock_return_ctx.new_data = mock_proc_data
    mock_return_ctx.coords = mock_proc_coords
    mock_return_ctx.bins = mock_bins_instance
    mock_return_ctx.overflows = 0
    mock_return_ctx.daily_log = mock_dlog
    mock_return_ctx.output_dict = {}
    mock_return_ctx.cached = None

    mock_run_day = mocker.patch(  # CAPTURE the mock object here
        "logic.src.pipeline.simulations.states.running.run_day", return_value=mock_return_ctx
    )

    # 8. Mock checkpointing
    mock_cp_instance = mocker.MagicMock()
    mock_cp_instance.load_state.return_value = (None, 0)  # Default: no resume
    mocker.patch(
        "logic.src.pipeline.simulations.states.initializing.SimulationCheckpoint",
        return_value=mock_cp_instance,
    )

    # Mock the context manager and its hook
    mock_hook = mocker.MagicMock()
    mock_cm = mocker.MagicMock()
    mock_cm.__enter__.return_value = mock_hook  # Yield the hook
    mocker.patch("logic.src.pipeline.simulations.states.running.checkpoint_manager", return_value=mock_cm)

    # 9. Mock utilities
    mock_log_to_json = mocker.MagicMock()
    mocker.patch("logic.src.pipeline.simulations.states.finishing.log_to_json", mock_log_to_json)
    mocker.patch("logic.src.pipeline.simulations.simulator.log_to_json", mock_log_to_json)
    mock_save_excel = mocker.patch("logic.src.pipeline.simulations.states.finishing.save_matrix_to_excel")
    mocker.patch("time.process_time", return_value=1.0)
    mocker.patch("pandas.DataFrame.to_excel")
    mocker.patch("statistics.mean", return_value=1.0)
    mocker.patch("statistics.stdev", return_value=0.1)

    def mock_tqdm_factory(*args, **kwargs):
        """
        Returns a mock instance that either iterates over the input (if it's an iterable)
        or just returns itself (if it's an int/total count).
        """
        if args and len(args) > 0:
            iterable = args[0]
        else:
            iterable = []

        mock_instance = mocker.MagicMock()
        mock_instance.update.return_value = None
        mock_instance.close.return_value = None

        if isinstance(iterable, (range, list, tuple)):
            mock_instance.__iter__.return_value = iter(iterable)
        else:
            pass

        mock_instance.__enter__.return_value = mock_instance
        mock_instance.__exit__.return_value = False

        return mock_instance

    mocker.patch(
        "logic.src.pipeline.simulations.states.running.tqdm",
        side_effect=mock_tqdm_factory,
        autospec=True,
    )

    # Return key mocks for modification in tests
    return {
        "checkpoint": mock_cp_instance,
        "hook": mock_hook,
        "run_day": mock_run_day,
        "log_to_json": mock_log_to_json,
        "save_excel": mock_save_excel,
        "bins": mock_bins_instance,
        "setup_model": mock_setup_model,
        "setup_env": mock_setup_env,
        "process_data": mock_process_data,
        "_setup_basedata": mock_setup_basedata,
        "_setup_dist_path_tup": mock_setup_dist,
    }


@pytest.fixture
def mock_load_dependencies(mocker):
    """
    Mocks all external dependencies:
    - os.path.join
    - pandas.read_csv
    - pandas.read_excel
    - the 'udef' module
    """
    # Mock os.path.join to simply concatenate with '/'
    mocker.patch("os.path.join", side_effect=lambda *args: "/".join(args))

    # Mock the file readers
    mock_read_csv = mocker.patch("pandas.read_csv")
    mock_read_excel = mocker.patch("pandas.read_excel")

    # Mock the 'udef' module
    mock_udef = MagicMock()
    mock_udef.WASTE_TYPES = {
        "paper": "Embalagens de papel e cartão",
        "plastic": "Mistura de embalagens",
    }
    mock_udef.LOCK_TIMEOUT = 30
    mocker.patch("logic.src.constants", mock_udef)

    return mock_read_csv, mock_read_excel, mock_udef


@pytest.fixture
def mock_run_day_deps(mocker):
    """
    Mocks all dependencies for the run_day function with REAL numpy arrays
    instead of MagicMocks to allow for math operations and slicing.
    """
    # 1. Setup Logic Data (Graph size 5)
    n_nodes = 5

    # Mock Bins object with real arrays
    mock_bins = MagicMock()
    mock_bins.is_stochastic.return_value = False
    mock_bins.loadFilling.return_value = (0, np.zeros(n_nodes), np.zeros(n_nodes), 0)
    mock_bins.stochasticFilling.return_value = (
        0,
        np.zeros(n_nodes),
        np.zeros(n_nodes),
        0,
    )
    mock_bins.c = np.full(n_nodes, 50.0)  # 50% fill
    mock_bins.means = np.full(n_nodes, 10.0)
    mock_bins.std = np.full(n_nodes, 1.0)
    mock_bins.collectlevl = np.full(n_nodes, 90.0)  # For last_minute policy
    mock_bins.n = n_nodes
    mock_bins.collect.return_value = (100.0, 2, 10.0, 5.0)

    # 2. Mock DataFrames
    mock_new_data = pd.DataFrame(
        {
            "ID": range(1, n_nodes + 1),
            "Stock": [0] * n_nodes,
            "Accum_Rate": [0] * n_nodes,
        }
    )

    # Create coordinates and set 'ID' as the index so .loc[id] works
    mock_coords = pd.DataFrame(
        {
            "ID": range(1, n_nodes + 1),
            "Lat": [40.0 + i * 0.1 for i in range(n_nodes)],
            "Lng": [-8.0 - i * 0.1 for i in range(n_nodes)],
        }
    )
    # We keep 'ID' as a column but also set it as index
    mock_coords.set_index("ID", drop=False, inplace=True)

    # 3. Real Numpy Distance Matrix (6x6: Depot=0, Nodes=1..5)
    matrix_size = n_nodes + 1
    real_dist_matrix = np.full((matrix_size, matrix_size), 10.0)
    np.fill_diagonal(real_dist_matrix, 0.0)

    # Int matrix for TSP
    real_distancesC = real_dist_matrix.astype(np.int32)

    mock_model_env = MagicMock()
    # Ensure graph is iterable: (edges, dist_matrix). dist_matrix must be a Tensor
    mock_graph = (MagicMock(), torch.tensor(real_dist_matrix))
    mock_model_ls = (MagicMock(), mock_graph, MagicMock())

    # 4. Construct distpath_tup with REAL arrays
    mock_dist_tup = (
        real_dist_matrix,  # Real float matrix
        MagicMock(),  # paths
        MagicMock(),  # dm_tensor
        real_distancesC,  # Real int matrix
    )

    mocker.patch("logic.src.pipeline.simulations.actions.logging.send_daily_output_to_gui")
    mocker.patch("logic.src.policies.single_vehicle.get_route_cost", return_value=50.0)
    mocker.patch("logic.src.policies.single_vehicle.find_route", return_value=[0, 1, 0])

    return {
        "bins": mock_bins,
        "model_env": mock_model_env,
        "model_ls": mock_model_ls,
        "distpath_tup": mock_dist_tup,
        "new_data": mock_new_data,
        "coords": mock_coords,
        "distance_matrix": real_dist_matrix,
        "distancesC": real_distancesC,
        "mock_send_output": mocker.patch("logic.src.pipeline.simulations.actions.logging.send_daily_output_to_gui"),
    }


@pytest.fixture
def make_day_context():
    """Fixture providing a factory to create a default SimulationDayContext with overrides."""

    def _make(**kwargs):
        full_policy = kwargs.get("full_policy", "policy_regular3_gamma1")

            # Determine policy_name and policy if not provided
        if "policy_name" not in kwargs or "policy" not in kwargs:
            # Simple policy (without gamma/threshold)
            p_simple = full_policy.split("_gamma")[0] if "_gamma" in full_policy else full_policy

            kwargs.setdefault("policy_name", p_simple)
            kwargs.setdefault("policy", p_simple)

        defaults = {
            "graph_size": 3,
            "full_policy": "policy_regular3_gamma1",
            "policy": "policy_regular3",
            "policy_name": "policy_regular3",
            "bins": MagicMock(),
            "new_data": MagicMock(),
            "coords": MagicMock(),
            "distance_matrix": MagicMock(),
            "distpath_tup": (None, None, None, None),
            "distancesC": MagicMock(),
            "paths_between_states": MagicMock(),
            "dm_tensor": MagicMock(),
            "run_tsp": True,
            "sample_id": 0,
            "overflows": 0,
            "day": 1,
            "model_env": MagicMock(),
            "model_ls": MagicMock(),
            "n_vehicles": 1,
            "area": "riomaior",
            "realtime_log_path": "mock_log.json",
            "waste_type": "paper",
            "current_collection_day": 1,
            "cached": None,
            "device": torch.device("cpu"),
            "lock": None,
            "hrl_manager": None,
            "gate_prob_threshold": 0.5,
            "mask_prob_threshold": 0.5,
            "two_opt_max_iter": 0,
            "config": {},
        }
        defaults.update(kwargs)
        return SimulationDayContext(**defaults)  # type: ignore

    return _make


@pytest.fixture
def mock_bins_params_loader(mocker):
    """
    Mock config loader for Bins to avoid file system reads.
    Returns: vehicle_capacity, revenue, density, expenses, bin_volume
    """
    return mocker.patch(
        "logic.src.pipeline.simulations.bins.base.load_area_and_waste_type_params",
        return_value=(None, 1.0, 10.0, 0.5, 1000.0)
    )


@pytest.fixture
def bins_stats(tmp_path, mock_bins_params_loader):
    """Create a Bins instance for stats testing."""
    return Bins(
        n=2,
        data_dir=str(tmp_path),
        sample_dist="gamma",
        area="test_area",
        waste_type="test_type",
    )


@pytest.fixture
def bins(tmp_path, mock_bins_params_loader):
    """Standard Bins fixture for general testing."""
    return Bins(
        n=10,
        data_dir=str(tmp_path),
        sample_dist="gamma",
        noise_mean=0.0,
        noise_variance=1.0  # Enable noise
    )
