"""
Shared pytest fixtures and configuration for all test modules.

This file is automatically loaded by pytest and provides fixtures
that can be used across all test files.
"""
import os
import sys
import torch
import shutil
import pytest
import tempfile
import numpy as np
import pandas as pd

from multiprocessing import Lock, Value
from pathlib import Path
from unittest.mock import MagicMock 

# The project root is THREE levels up from conftest.py:
# conftest.py -> test -> logic -> WSmart-Route (Project Root)
project_root = Path(__file__).resolve().parent.parent.parent

# Add the project root to sys.path. This allows 'import logic.src...' 
# to resolve 'logic' as a package within WSmart-Route/.
sys.path.insert(0, str(project_root))

from logic.src.pipeline.simulator.bins import Bins
from logic.src.pipeline.reinforcement_learning.meta.contextual_bandits import WeightContextualBandit
from logic.src.pipeline.reinforcement_learning.meta.multi_objective import MORLWeightOptimizer
from logic.src.pipeline.reinforcement_learning.meta.temporal_difference_learning import CostWeightManager
from logic.src.pipeline.reinforcement_learning.meta.weight_optimizer import RewardWeightOptimizer
from logic.src.models.attention_model import AttentionModel
from logic.src.models.gat_lstm_manager import GATLSTManager
from logic.src.models.temporal_am import TemporalAttentionModel
from logic.src.models.model_factory import NeuralComponentFactory
from logic.src.models.subnets.attention_decoder import AttentionDecoder


# ============================================================================
# Shared Fixtures
# ============================================================================

@pytest.fixture
def base_train_args():
    """Fixture providing base training arguments"""
    return [
        'script.py', 'train',
        '--problem', 'vrpp',
        '--graph_size', '20',
        '--batch_size', '256',
        '--epoch_size', '128000',
        '--n_epochs', '25'
    ]


@pytest.fixture
def base_mrl_args():
    """Fixture providing base MRL training arguments"""
    return [
        'script.py', 'mrl_train',
        '--problem', 'vrpp',
        '--batch_size', '256',
        '--epoch_size', '128000',
        '--mrl_method', 'cb'
    ]


@pytest.fixture
def base_hp_optim_args():
    """Fixture providing base hyperparameter optimization arguments"""
    return [
        'script.py', 'hp_optim',
        '--batch_size', '256',
        '--epoch_size', '128000',
        '--hop_method', 'bo'
    ]


@pytest.fixture
def base_gen_data_args():
    """Fixture providing base data generation arguments"""
    return [
        'script.py', 'gen_data',
        '--problem', 'vrpp',
        '--dataset_size', '10000',
        '--graph_sizes', '20'
    ]


@pytest.fixture
def base_eval_args():
    """Fixture providing base evaluation arguments"""
    return [
        'script.py', 'eval',
        '--datasets', 'dataset1.pkl',
        '--model', 'model.pt'
    ]


@pytest.fixture
def base_test_args():
    """Fixture providing base simulator test arguments"""
    return [
        'script.py', 'test_sim',  # Changed from 'test' to 'test_sim' to match your parser
        '--policies', 'policy1',
        '--days', '31',
        '--size', '50'
    ]


@pytest.fixture
def base_file_system_update_args():
    """Fixture providing base file system update arguments"""
    return [
        'script.py', 'file_system', 'update',
        '--target_entry', 'path/to/file.pkl'
    ]


@pytest.fixture
def base_file_system_delete_args():
    """Fixture providing base file system delete arguments"""
    return [
        'script.py', 'file_system', 'delete'
    ]


@pytest.fixture
def base_file_system_crypto_args():
    """Fixture providing base file system cryptography arguments"""
    return [
        'script.py', 'file_system', 'cryptography'
    ]


@pytest.fixture
def base_gui_args():
    """Fixture providing base GUI arguments"""
    return [
        'script.py', 'gui'
    ]


# Define standard opts (arguments) for gen_data command
@pytest.fixture
def gen_data_opts():
    """Returns a basic set of mock arguments (opts) for generate_datasets."""
    return {
        'name': 'test_suite',
        'problem': 'vrpp',
        'dataset_size': 100,
        'graph_sizes': [20],
        'data_distributions': ['all'],
        'area': 'SomeArea',
        'waste_type': 'SomeWaste',
        'focus_graphs': ['graph.csv'],
        'focus_size': 0,
        'vertex_method': 'Uniform',
        'is_gaussian': False,
        'sigma': 0.1,
        'seed': 42,
        'dataset_type': 'train',
        'n_epochs': 1,
        'epoch_start': 0,
        'data_dir': 'datasets',
        'filename': 'test',
        'f': True,
        'penalty_factor': 10 
    }


# Define standard opts (arguments) for test_sim command
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
        'days': 10,
        'size': 2,
        'area': 'riomaior',
        'waste_type': 'paper',
        'policies': ['test_policy_gamma1'],
        'output_dir': 'test_output',
        'checkpoint_dir': 'temp',
        'resume': False,
        'distance_method': 'hsd',
        'dm_filepath': None,
        'env_file': None,
        'gapik_file': None,
        'symkey_name': None,
        'edge_threshold': 50,
        'edge_method': 'knn',
        'temperature': 1.0,
        'decode_type': 'greedy',
        'vertex_method': 'mmn',
        'server_run': False,
        'gplic_file': None,
        'cache_regular': False,
        'waste_filepath': None,
        'run_tsp': False,
        'n_vehicles': 1,
        'checkpoint_days': 5,
        'no_progress_bar': True,
        'data_distribution': 'gamma',
        'n_samples': 1,
        'seed': 42,
        'problem': 'vrpp',
        'stats_filepath': None,
        'model_path': None,
    }

# --- Fixtures for Policy Unit Tests ---
@pytest.fixture
def policy_deps(mocker):
    """
    Provides a comprehensive set of mocks for unit-testing individual policy functions.
    This is different from conftest.py's mock_run_day_deps, which mocks all
    dependencies for the `run_day` function itself.
    """
    
    # 1. Mock common data (5 bins + 1 depot)
    # Bins 1-5 (indices 0-4)
    bins_waste = np.array([10.0, 95.0, 30.0, 85.0, 50.0]) 
    
    # Distances (Depot=0, Bins=1-5)
    distancesC = np.array([
        [0, 10, 10, 15, 20, 15], # Depot 0
        [10, 0, 5, 10, 15, 10], # Bin 1
        [10, 5, 0, 10, 15, 10], # Bin 2
        [15, 10, 10, 0, 5, 5],  # Bin 3
        [20, 15, 15, 5, 0, 5],  # Bin 4
        [15, 10, 10, 5, 5, 0]   # Bin 5
    ], dtype=np.int32)
    
    # Paths for policy_last_minute_and_path (Nodes 0-5)
    paths_between_states = [
        [[]], # 0
        [[1, 0], [], [1, 2], [1, 5, 3], [1, 5, 4], [1, 5]], # 1
        [[2, 0], [2, 1], [], [2, 1, 5, 3], [2, 1, 5, 4], [2, 1, 5]], # 2
        [[3, 5, 0], [3, 5, 1], [3, 5, 1, 2], [], [3, 4], [3, 5]], # 3
        [[4, 5, 0], [4, 5, 1], [4, 5, 1, 2], [4, 3], [], [4, 5]], # 4
        [[5, 0], [5, 1], [5, 1, 2], [5, 3], [5, 4], []]  # 5
    ]
    
    # 2. Mock dependent functions
    mock_load_params = mocker.patch(
        'logic.src.pipeline.simulator.loader.load_area_and_waste_type_params',
        return_value=(4000, 0.16, 21.0, 1.0, 2.5) # Q, R, B, C, V
    )
    mocker.patch('logic.src.policies.regular.load_area_and_waste_type_params', mock_load_params)
    mocker.patch('logic.src.policies.last_minute.load_area_and_waste_type_params', mock_load_params)
    
    # Mock TSP solver (used by last_minute and regular)
    mock_find_route = mocker.patch(
        'logic.src.policies.single_vehicle.find_route', 
        return_value=[0, 1, 3, 0] # Default mock tour
    )
    mocker.patch('logic.src.policies.regular.find_route', mock_find_route)
    mocker.patch('logic.src.policies.last_minute.find_route', mock_find_route)
    
    # Mock multi-tour splitter
    mock_get_multi_tour = mocker.patch(
        'logic.src.policies.single_vehicle.get_multi_tour',
        side_effect=lambda tour, *args: tour # Pass-through
    )
    mocker.patch('logic.src.policies.regular.get_multi_tour', mock_get_multi_tour)
    mocker.patch('logic.src.policies.last_minute.get_multi_tour', mock_get_multi_tour)

    return {
        "n_bins": 5,
        "bins_waste": bins_waste,
        "distancesC": distancesC,
        "paths_between_states": paths_between_states,
        "mocks": {
            "load_params": mock_load_params,
            "find_route": mock_find_route,
            "get_multi_tour": mock_get_multi_tour
        }
    }

# ============================================================================
# Basic Class Fixtures
# ============================================================================

@pytest.fixture
def basic_bins(tmp_path):
    """Returns a basic Bins instance for testing."""
    return Bins(n=10, data_dir=str(tmp_path), sample_dist="gamma", area="riomaior", waste_type="paper")


@pytest.fixture
def basic_checkpoint(mocker, tmp_path):
    """
    Sets up a mocked environment for SimulationCheckpoint testing.
    - Mocks ROOT_DIR to a temporary path.
    - Mocks os.listdir to prevent FileNotFoundError.
    """
    
    # 1. Mock ROOT_DIR to point to a temporary path
    mock_root = tmp_path / "mock_root"
    mocker.patch(
        'logic.src.pipeline.simulator.checkpoints.ROOT_DIR', 
        new=str(mock_root)
    )

    # 2. Define the path structure passed to the constructor
    # The test implies the output_dir input ends with "results"
    mock_output_dir_base = tmp_path / "test_assets" / "results"
    
    # 3. Mock os.listdir since it's used by find_last_checkpoint_day() (called by get_checkpoint_file without 'day')
    mocker.patch('os.listdir', return_value=[])

    # 4. Initialize SimulationCheckpoint
    from logic.src.pipeline.simulator.checkpoints import SimulationCheckpoint
    cp = SimulationCheckpoint(
        # Pass the path that ends in 'results'
        output_dir=str(mock_output_dir_base),
        checkpoint_dir="temp",
        policy="test_policy",
        sample_id=1
    )
    
    # Ensure the checkpoint directories exist for later tests (like save/load)
    os.makedirs(cp.checkpoint_dir, exist_ok=True)
    os.makedirs(cp.output_dir, exist_ok=True)
    
    return cp

# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

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


# ============================================================================
# File Creation Fixtures
# ============================================================================

@pytest.fixture
def sample_dataset_file(temp_data_dir):
    """Create a sample dataset file"""
    dataset_path = Path(temp_data_dir) / "sample_dataset.pkl"
    # Create an empty file for testing
    dataset_path.touch()
    return str(dataset_path)


@pytest.fixture
def sample_model_file(temp_model_dir):
    """Create a sample model file"""
    model_path = Path(temp_model_dir) / "sample_model.pt"
    # Create an empty file for testing
    model_path.touch()
    return str(model_path)


@pytest.fixture
def sample_config_file(temp_data_dir):
    """Create a sample configuration file"""
    config_path = Path(temp_data_dir) / "sample_config.json"
    # Create an empty file for testing
    config_path.touch()
    return str(config_path)


# ============================================================================
# Mock Fixtures
# ============================================================================

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


# Mock problem generation functions and data utilities
@pytest.fixture(autouse=True)
def mock_dependencies(mocker):
    """Mocks all external data generation and saving dependencies."""
    
    # 1. Mock the specific problem generators imported into generate_data.py
    # Since they are imported via `from .generate_problem_data import *`, we mock 
    # them as attributes of the generate_data module.

    # Mocks removed as these functions are no longer imported in generate_data.py
    # and have been replaced by VRPInstanceBuilder.
    # mocker.patch('logic.src.data.generate_data.generate_vrpp_data', return_value=[(None, None)])
    # mocker.patch('logic.src.data.generate_data.generate_wcvrp_data', return_value=[(None, None)])
    # mocker.patch('logic.src.data.generate_data.generate_wsr_data', return_value=[(None, None)])
    
    # 3. Mock the save utility
    mocker.patch('logic.src.data.generate_data.save_dataset', return_value=None)


# Mocks for WSmart+ Route simulator
@pytest.fixture
def mock_data_dir(tmp_path):
    """Creates a mock data directory structure."""
    data_dir = tmp_path / "data" / "wsr_simulator"
    
    # For loader.py (load_depot)
    coord_dir = data_dir / "coordinates"
    coord_dir.mkdir(parents=True, exist_ok=True)
    facilities_data = {
        'Sigla': ['RM', 'OTHER'],
        'Lat': [40.0, 41.0],
        'Lng': [-8.0, -9.0],
        'ID': [1000, 1001]
    }
    pd.DataFrame(facilities_data).to_csv(coord_dir / "Facilities.csv", index=False)
    
    # For loader.py (load_simulator_data)
    pd.DataFrame({'ID': [1, 2, 3], 'Lat': [40.1, 40.2, 40.3], 'Lng': [-8.1, -8.2, -8.3]}).to_csv(
        coord_dir / "old_out_info[riomaior].csv", index=False
    )
    
    # For loader.py (load_simulator_data)
    data_file = data_dir / "daily_c[riomaior_paper].csv"
    sim_data = {
        'Date': ['2023-01-01', '2023-01-02'],
        '1': [10, 20],
        '2': [15, 25],
        '3': [20, 30]
    }
    pd.DataFrame(sim_data).to_csv(data_file, index=False)

    return data_dir


@pytest.fixture
def mock_coords_df():
    """Returns a mock coordinates DataFrame."""
    depot = pd.DataFrame({'ID': [0], 'Lat': [40.0], 'Lng': [-8.0], 'Stock': [0], 'Accum_Rate': [0]})
    depot = depot.set_index(pd.Index([0]))
    
    coords = pd.DataFrame({
        'ID': [1, 2],
        'Lat': [40.1, 40.2],
        'Lng': [-8.1, -8.2]
    })
    return depot, coords


@pytest.fixture
def mock_data_df():
    """Returns a mock data DataFrame."""
    data = pd.DataFrame({
        'ID': [1, 2],
        'Stock': [10, 20],
        'Accum_Rate': [5, 3]
    })
    return data


@pytest.fixture
def mock_lock_counter(mocker):
    """
    Provides mock multiprocessing.Lock and Value objects.
    """
    mock_lock = mocker.MagicMock(spec=Lock())
    mock_counter = mocker.MagicMock(spec=Value('i'))
    mock_counter.value = 0
    # Mock the context manager for the lock
    mock_counter_lock = mocker.MagicMock()
    mock_counter_lock.__enter__.return_value = None
    mock_counter_lock.__exit__.return_value = None
    return mock_lock, mock_counter


@pytest.fixture
def mock_torch_device():
    """Returns a CPU torch device."""
    return torch.device('cpu')


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
    return mock_bins


@pytest.fixture
def mock_sim_dependencies(mocker, tmp_path, mock_bins_instance):
    """
    Mocks all major external dependencies for single_simulation
    and sequential_simulations.
    """
    # 1. Patch ROOT_DIR in both modules to ensure consistency
    mocker.patch('logic.src.pipeline.simulator.states.ROOT_DIR', str(tmp_path))
    mocker.patch('logic.src.pipeline.simulator.simulation.ROOT_DIR', str(tmp_path))
    
    # 2. Mock loader functions
    mock_depot = pd.DataFrame({'ID': [0], 'Lat': [40], 'Lng': [-8], 'Stock': [0], 'Accum_Rate': [0]})
    mock_data = pd.DataFrame({'ID': [1, 2], 'Stock': [10, 20], 'Accum_Rate': [0.1, 0.2]})
    mock_coords = pd.DataFrame({'ID': [1, 2], 'Lat': [40.1, 40.2], 'Lng': [-8.1, -8.2]})
    mocker.patch('logic.src.pipeline.simulator.processor.load_depot', return_value=mock_depot)
    mocker.patch('logic.src.pipeline.simulator.processor.load_simulator_data', 
                 return_value=(mock_data.copy(), mock_coords.copy()))

    # Mock setup_basedata in states to return the tuple directly
    mock_setup_basedata = mocker.patch('logic.src.pipeline.simulator.states.setup_basedata',
                 return_value=(mock_data.copy(), mock_coords.copy(), mock_depot.copy()))

    # 3. Mock processor functions (patched in states where imported)
    mock_proc_data = pd.DataFrame({'ID': [0, 1, 2], 'Stock': [0, 10, 20]})
    mock_proc_coords = pd.DataFrame({'ID': [0, 1, 2], 'Lat': [40, 40.1, 40.2], 'Lng': [-8, -8.1, -8.2]})
    mock_process_data = mocker.patch('logic.src.pipeline.simulator.states.process_data', 
                 return_value=(mock_proc_data.copy(), mock_proc_coords.copy()))
    mocker.patch('logic.src.pipeline.simulator.states.process_model_data', 
                 return_value=('mock_model_tup_0', 'mock_model_tup_1'))

    # 4. Mock network functions (patched in states where imported)
    mock_dist_tup = (np.array([[0, 1], [1, 0]]), 'mock_paths', 'mock_tensor', 'mock_distC')
    mock_adj_matrix = np.array([[1, 1], [1, 1]])
    mock_setup_dist = mocker.patch('logic.src.pipeline.simulator.states.setup_dist_path_tup', 
                 return_value=(mock_dist_tup, mock_adj_matrix))
    # Still patch processor lower level functions if used elsewhere or implicitly? 
    # But states.setup_dist_path_tup mock prevents calling them.
    mocker.patch('logic.src.pipeline.simulator.processor.compute_distance_matrix', 
                 return_value=np.array([[0,1],[1,0]]))
    mocker.patch('logic.src.pipeline.simulator.processor.apply_edges', 
                 return_value=('mock_dist_edges', 'mock_paths', 'mock_adj'))
    mocker.patch('logic.src.pipeline.simulator.processor.get_paths_between_states', 
                 return_value='mock_all_paths')

    # 5. Mock Bins class
    mocker.patch('logic.src.pipeline.simulator.states.Bins', return_value=mock_bins_instance)

    # 6. Mock setup functions
    mock_setup_model = mocker.patch(
        'logic.src.pipeline.simulator.states.setup_model',
        return_value=(MagicMock(), MagicMock()) 
    )
    mock_setup_env = mocker.patch('logic.src.utils.setup_utils.setup_env', 
                 return_value='mock_or_env')

    # 7. Mock day function
    mock_dlog = {'day': 1, 'overflows': 0, 'kg_lost': 0, 'kg': 0, 'ncol': 0, 'km': 0, 'kg/km': 0, 'tour': [0]}
    mock_data_ls = (mock_proc_data, mock_proc_coords, mock_bins_instance)
    mock_output_ls = (0, mock_dlog, {}) # overflows, dlog, output_dict
    mock_run_day = mocker.patch( # CAPTURE the mock object here
        'logic.src.pipeline.simulator.states.run_day', 
        return_value=(mock_data_ls, mock_output_ls, None)
    )

    # 8. Mock checkpointing
    mock_cp_instance = mocker.MagicMock()
    mock_cp_instance.load_state.return_value = (None, 0) # Default: no resume
    mocker.patch('logic.src.pipeline.simulator.states.SimulationCheckpoint', 
                 return_value=mock_cp_instance)
    
    # Mock the context manager and its hook
    mock_hook = mocker.MagicMock()
    mock_cm = mocker.MagicMock()
    mock_cm.__enter__.return_value = mock_hook # Yield the hook
    mocker.patch('logic.src.pipeline.simulator.states.checkpoint_manager', 
                return_value=mock_cm)

    # 9. Mock utilities
    mock_log_to_json = mocker.MagicMock()
    mocker.patch('logic.src.pipeline.simulator.states.log_to_json', mock_log_to_json)
    mocker.patch('logic.src.pipeline.simulator.simulation.log_to_json', mock_log_to_json)
    # output_stats removed
    mock_save_excel = mocker.patch('logic.src.pipeline.simulator.states.save_matrix_to_excel')
    mocker.patch('time.process_time', return_value=1.0)
    # Removed os.makedirs patch to allow directory creation (since ROOT_DIR is tmp)
    mocker.patch('pandas.DataFrame.to_excel')
    mocker.patch('statistics.mean', return_value=1.0)
    mocker.patch('statistics.stdev', return_value=0.1)

    mock_tqdm_instance = mocker.MagicMock()
    mock_tqdm_instance.update.return_value = None
    mock_tqdm_instance.close.return_value = None

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
        
        # Crucial check: Only make the mock iterable if the input argument is actually iterable (e.g., a range or list).
        if isinstance(iterable, (range, list, tuple)):
            # Configure the mock to yield the values from the actual iterable
            mock_instance.__iter__.return_value = iter(iterable)
        else:
            # If it's an integer (like opts['days'] = 5), we assume it's the total count for a manual bar.
            # MagicMock's default __iter__ behavior is fine for a non-loop bar, or we explicitly remove it.
            # Since MagicMock often fails as an iterator if __iter__ isn't set, we just skip setting the return_value.
            pass

        # Configure as a context manager (if used in 'with')
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.__exit__.return_value = False
        
        return mock_instance

    mocker.patch(
        'logic.src.pipeline.simulator.states.tqdm', 
        side_effect=mock_tqdm_factory, # Use side_effect to dynamically return the iterable mock
        autospec=True
    )

    # Return key mocks for modification in tests
    return {
        'checkpoint': mock_cp_instance,
        'hook': mock_hook,
        'run_day': mock_run_day,
        'log_to_json': mock_log_to_json,
        'save_excel': mock_save_excel,
        'bins': mock_bins_instance,
        'setup_model': mock_setup_model,
        'setup_env': mock_setup_env,
        'process_data': mock_process_data,
        '_setup_basedata': mock_setup_basedata,
        '_setup_dist_path_tup': mock_setup_dist,
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
    mocker.patch('os.path.join', side_effect=lambda *args: '/'.join(args))
    
    # Mock the file readers
    mock_read_csv = mocker.patch('pandas.read_csv')
    mock_read_excel = mocker.patch('pandas.read_excel')
    
    # Mock the 'udef' module
    # We must mock it where it's *used*, e.g., 'logic.src.utils.definitions'
    mock_udef = MagicMock()
    mock_udef.WASTE_TYPES = {
        'paper': 'Embalagens de papel e cartão',
        'plastic': 'Mistura de embalagens'
    }
    mock_udef.LOCK_TIMEOUT = 30
    mocker.patch('logic.src.utils.definitions', mock_udef)
    
    # Return the mocks so tests can configure them
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
    mock_bins.stochasticFilling.return_value = (0, np.zeros(n_nodes), np.zeros(n_nodes), 0)
    mock_bins.c = np.full(n_nodes, 50.0) # 50% fill
    mock_bins.means = np.full(n_nodes, 10.0)
    mock_bins.std = np.full(n_nodes, 1.0)
    mock_bins.collectlevl = np.full(n_nodes, 90.0) # For last_minute policy
    mock_bins.n = n_nodes
    mock_bins.collect.return_value = (100.0, 2, 10.0, 5.0) 

    # 2. Mock DataFrames
    mock_new_data = pd.DataFrame({
        'ID': range(1, n_nodes + 1), 
        'Stock': [0]*n_nodes, 
        'Accum_Rate': [0]*n_nodes
    })
    
    # Create coordinates and set 'ID' as the index so .loc[id] works
    mock_coords = pd.DataFrame({
        'ID': range(1, n_nodes + 1),
        'Lat': [40.0 + i*0.1 for i in range(n_nodes)],
        'Lng': [-8.0 - i*0.1 for i in range(n_nodes)]
    })
    # We keep 'ID' as a column but also set it as index
    mock_coords.set_index('ID', drop=False, inplace=True)

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
        real_dist_matrix,    # Real float matrix
        MagicMock(),         # paths
        MagicMock(),         # dm_tensor
        real_distancesC      # Real int matrix
    )
    
    # 5. Patch external calls
    # Note: These patches affect the 'definition' modules. 
    # Tests in test_policies.py must patch the 'usage' module (day.py).
    mock_policy_regular = mocker.patch(
        'logic.src.policies.regular.policy_regular', return_value=[0, 1, 2, 0] 
    )
    mock_send_output = mocker.patch(
        'logic.src.pipeline.simulator.actions.send_daily_output_to_gui', autospec=True
    )
    mocker.patch('logic.src.policies.single_vehicle.get_route_cost', return_value=50.0)
    mocker.patch('logic.src.policies.single_vehicle.find_route', return_value=[0, 1, 0])

    return {
        'bins': mock_bins,
        'model_env': mock_model_env,
        'model_ls': mock_model_ls,
        'distpath_tup': mock_dist_tup,
        'new_data': mock_new_data,
        'coords': mock_coords,
        'mock_policy_regular': mock_policy_regular,
        'mock_send_output': mock_send_output,
    }


@pytest.fixture
def mock_policy_common_data():
    """Provides common data structures (distances, waste, paths) for policy unit tests."""
    # 5 bins + 1 depot (node 0)
    distancesC = np.array([
        [0, 10, 10, 15, 20, 15], # Depot 0
        [10, 0, 5, 10, 15, 10], # Bin 1 (idx 1)
        [10, 5, 0, 10, 15, 10], # Bin 2 (idx 2)
        [15, 10, 10, 0, 5, 5],  # Bin 3 (idx 3)
        [20, 15, 15, 5, 0, 5],  # Bin 4 (idx 4)
        [15, 10, 10, 5, 5, 0]   # Bin 5 (idx 5)
    ], dtype=np.int32)
    
    # Fill levels for bins 1-5 (indices 0-4)
    bins_waste = np.array([10.0, 95.0, 30.0, 85.0, 50.0])
    
    # Mock paths for 'last_minute_and_path' testing (full 6x6 node structure)
    paths_between_states = [
        [[]] * 6, 
        [[]] * 6, 
        [[2, 0], [2, 1], [2], [2, 1, 5, 3], [2, 1, 5, 4], [2, 1, 5]], # Example paths
        [[]] * 6,
        [[]] * 6,
        [[]] * 6,
    ]
    
    return {
        "n_bins": 5,
        "bins_waste": bins_waste,
        "distancesC": distancesC,
        "distance_matrix": distancesC.astype(float),
        "paths_between_states": paths_between_states
    }


@pytest.fixture(autouse=True)
def mock_policy_dependencies(mocker):
    """Mocks common policy dependencies (loader, solver) for unit tests."""
    # Mock TSP solver (used by last_minute)
    mocker.patch(
        'logic.src.policies.single_vehicle.find_route', 
        return_value=[0, 1, 3, 0] # Default mock tour for 2 bins
    )
    # Mock multi-tour splitter
    mocker.patch(
        'logic.src.policies.single_vehicle.get_multi_tour',
        side_effect=lambda tour, *args: tour # Pass-through
    )
    # Mock distance matrix used by single_vehicle helpers
    mocker.patch(
        'logic.src.policies.single_vehicle.get_route_cost',
        return_value=50.0 
    )


@pytest.fixture(autouse=True)
def mock_lookahead_aux(mocker):
    """Mocks the internal look_ahead_aux dependencies."""
    
    # Mocks must be autospec=True to be bound correctly to the look_ahead module import
    mocker.patch(
        'logic.src.policies.look_ahead.should_bin_be_collected',
        autospec=True,
        side_effect=lambda fill, rate: fill + rate >= 100
    )
    mocker.patch(
        'logic.src.policies.look_ahead.update_fill_levels_after_first_collection',
        autospec=True,
        # Returns fill levels after collected bins are reset
        return_value=np.array([0.0, 10.0, 30.0, 40.0, 50.0]) 
    )
    mocker.patch(
        'logic.src.policies.look_ahead.get_next_collection_day',
        autospec=True,
        return_value=5 # Mocked next overflow day
    )
    mocker.patch(
        'logic.src.policies.look_ahead.add_bins_to_collect',
        autospec=True,
        return_value=[0, 1, 2] # Mocked final bin list
    )


@pytest.fixture
def mock_vrpp_inputs(mock_policy_common_data):
    """Provides data structures needed for VRPP policies."""
    data = mock_policy_common_data
    
    # Mock predicted values (media + param * std)
    media = np.full(data['n_bins'], 10.0)
    std = np.full(data['n_bins'], 1.0)
    
    return {
        "bins": data['bins_waste'], # [10.0, 95.0, 30.0, 85.0, 50.0]
        "distances": data['distance_matrix'].tolist(), # Use float matrix
        "distance_matrix": data['distance_matrix'],
        "media": media,
        "std": std,
        "must_go_bins": [1, 3], # Bin indices (0-indexed)
        "binsids": list(range(data['n_bins'])), # [0, 1, 2, 3, 4]
        }


@pytest.fixture
def mock_optimizer_data(mock_policy_common_data):
    """Provides data structures needed for VRPP policies."""
    data = mock_policy_common_data
    
    # Mock predicted values (media + param * std)
    media = np.full(data['n_bins'], 10.0)
    std = np.full(data['n_bins'], 1.0)
    
    return {
        "bins": data['bins_waste'], # [10.0, 95.0, 30.0, 85.0, 50.0]
        "distances": data['distance_matrix'].tolist(), # Use float matrix
        "media": media,
        "std": std
    }


@pytest.fixture
def hgs_inputs():
    """Provide standard inputs for HGS tests."""
    # Simple scenario: Depot + 4 nodes
    # Distances: linear 0--1--2--3--4
    dist_matrix = [
        [0, 10, 20, 30, 40],
        [10, 0, 10, 20, 30],
        [20, 10, 0, 10, 20],
        [30, 20, 10, 0, 10],
        [40, 30, 20, 10, 0]
    ]

    demands = {1: 10, 2: 10, 3: 10, 4: 10}
    capacity = 100
    R = 1.0
    C = 1.0

    global_must_go = {2, 4}  # Must go bins
    local_to_global = {0: 1, 1: 2, 2: 3, 3: 4}  # Linear mapping

    vrpp_tour_global = [2, 4, 1, 3]  # Some initial tour

    return dist_matrix, demands, capacity, R, C, global_must_go, local_to_global, vrpp_tour_global

# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically setup test environment for all tests"""
    # Set test mode environment variable
    original_test_mode = os.environ.get('TEST_MODE')
    os.environ['TEST_MODE'] = 'true'
    
    yield
    
    # Restore original environment
    if original_test_mode is None:
        os.environ.pop('TEST_MODE', None)
    else:
        os.environ['TEST_MODE'] = original_test_mode


@pytest.fixture
def disable_wandb():
    """Disable wandb logging for tests"""
    import os
    original_wandb_mode = os.environ.get('WANDB_MODE')
    os.environ['WANDB_MODE'] = 'disabled'
    
    yield
    
    if original_wandb_mode is None:
        os.environ.pop('WANDB_MODE', None)
    else:
        os.environ['WANDB_MODE'] = original_wandb_mode


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_root(request):
    """Cleanup test root directory after session."""
    def finalizer():
        """Finalizer to clean up test output."""
        path_to_clean = Path("assets/test_output")
        if path_to_clean.exists():
            shutil.rmtree(path_to_clean)
    request.addfinalizer(finalizer)


@pytest.fixture
def mock_train_model(mocker, mock_torch_device):
    """
    Returns a mock PyTorch model for training tests.
    It mimics the signature: cost, log_likelihood, c_dict, pi = model(x, cost_weights, return_pi, pad)
    """
    mock_model = mocker.MagicMock(spec=torch.nn.Module)
    
    # Setup standard return values
    # cost: (batch_size,) tensor
    # log_likelihood: (batch_size,) tensor
    # c_dict: dict of tensors
    # pi: (batch_size, seq_len)
    
    cost = torch.tensor([10.0, 12.0], device=mock_torch_device, requires_grad=True)
    log_likelihood = torch.tensor([-0.5, -0.6], device=mock_torch_device, requires_grad=True)
    c_dict = {
        'length': torch.tensor([5.0, 6.0], device=mock_torch_device),
        'waste': torch.tensor([50.0, 60.0], device=mock_torch_device),
        'overflows': torch.tensor([0.0, 1.0], device=mock_torch_device)
    }
    pi = torch.tensor([[0, 1, 0], [0, 2, 0]], device=mock_torch_device)
    
    # Configure return values on instance call
    mock_model.return_value = (cost, log_likelihood, c_dict, pi)
    
    # Allow .to(device) chaining
    mock_model.to.return_value = mock_model
    
    # DataParallel support mocks
    mock_model.module = mock_model
    
    # Mock set_decode_type method
    mock_model.set_decode_type = mocker.MagicMock()
    
    return mock_model


@pytest.fixture
def mock_optimizer(mocker):
    """Returns a mock optimizer."""
    mock_opt = mocker.MagicMock()
    mock_opt.param_groups = [{'params': [], 'lr': 0.001}]
    return mock_opt


@pytest.fixture
def mock_baseline(mocker):
    """Returns a mock baseline."""
    mock_bl = mocker.MagicMock()
    mock_bl.wrap_dataset.side_effect = lambda x: x # Pass through
    mock_bl.unwrap_batch.side_effect = lambda x: (x, None) # Simple unwrap
    # eval returns (bl_val, bl_loss)
    mock_bl.eval.return_value = (torch.zeros(2), 0.0) 
    return mock_bl


# ============================================================================
# HPO Fixtures (Moved from test_hp_optim.py)
# ============================================================================

@pytest.fixture
def hpo_opts():
    """Provide standard HPO options."""
    return {
        'problem': 'wcvrp',
        'graph_size': 20,
        'save_dir': 'test_save_dir',
        'load_path': 'test_load_path',
        'resume': False,
        'val_size': 10,
        'val_dataset': None,
        'area': 'Rio Maior',
        'waste_type': 'test_waste',
        'distance_method': 'test_dist',
        'data_distribution': 'test_dist',
        'vertex_method': 'test_vertex',
        'edge_threshold': 10,
        'edge_method': 'test_edge',
        'focus_graph': None,
        'eval_focus_size': 0,
        'dm_filepath': None,
        'enable_scaler': False,
        'no_cuda': True,
        'train_time': False,
        'epoch_start': 0,
        'n_epochs': 1,
        'eval_batch_size': 2,
        'no_progress_bar': True,
        'hop_range': [0.1, 1.0],
        'hop_epochs': 1,
        'eta': 3,
        'indpb': 0.1,
        'tournsize': 3,
        'n_pop': 5,
        'cxpb': 0.5,
        'mutpb': 0.2,
        'n_gen': 1,
        'verbose': 0,
        'n_startup_trials': 1,
        'n_warmup_steps': 1,
        'interval_steps': 1,
        'logging': {
            'log_output': False
        },
        'seed': 1234,
        'run_name': 'test_run',
        'metric': 'loss',
        'cpu_cores': 1,
        'log_dir': 'test_log_dir',
        'num_samples': 1,
        'max_tres': 1,
        'max_conc': 1,
        'no_tensorboard': True,
        'device': 'cpu'
    }

# ============================================================================
# MRL Fixtures (Moved from test_mrl_train.py)
# ============================================================================

@pytest.fixture
def bandit_setup():
    """Setup WeightContextualBandit instance."""
    initial_weights = {'w_waste': 1.0, 'w_over': 1.0}
    weight_ranges = {'w_waste': (0.1, 5.0), 'w_over': (0.1, 5.0)}
    context_features = ['waste', 'overflow', 'day']
    dummy_dist_matrix = torch.rand(10, 2)
    
    bandit = WeightContextualBandit(
        num_days=10,
        distance_matrix=dummy_dist_matrix,
        initial_weights=initial_weights,
        context_features=context_features,
        features_aggregation='avg',
        exploration_strategy='epsilon_greedy',
        exploration_factor=0.5,
        num_weight_configs=5,
        weight_ranges=weight_ranges,
        window_size=5
    )
    return bandit

@pytest.fixture
def morl_setup():
    """Setup MORLWeightOptimizer instance."""
    initial_weights = {'w_waste': 1.0, 'w_over': 1.0, 'w_len': 1.0}
    optimizer = MORLWeightOptimizer(
        initial_weights=initial_weights,
        weight_names=['w_waste', 'w_over', 'w_len'],
        objective_names=['waste_efficiency', 'overflow_rate'],
        weight_ranges=[0.01, 5.0],
        history_window=10,
        exploration_factor=0.2, 
        adaptation_rate=0.1
    )
    return optimizer

@pytest.fixture
def cwm_setup():
    """Setup CostWeightManager instance."""
    initial_weights = {'waste': 1.0, 'over': 1.0, 'len': 1.0}
    manager = CostWeightManager(
        initial_weights=initial_weights,
        learning_rate=0.1,
        decay_rate=0.9,
        weight_ranges=[0.1, 5.0],
        window_size=5
    )
    return manager

@pytest.fixture
def rwo_setup():
    """Setup RewardWeightOptimizer instance."""
    # Helper mock class for RWO
    class MockModel(torch.nn.Module):
        """Mock Neural Model for RewardWeightOptimizer."""
        def __init__(self, input_size, hidden_size, output_size):
            """Initialize mock model."""
            super().__init__()
            self.layer = torch.nn.Linear(input_size, output_size)
        def forward(self, x):
            """Mock forward pass."""
            return self.layer(x[:, -1, :]), None 

    initial_weights = {'w1': 1.0, 'w2': 1.0}
    
    optimizer = RewardWeightOptimizer(
        model_class=MockModel,
        initial_weights=initial_weights,
        history_length=5,
        hidden_size=10,
        lr=0.01,
        device='cpu',
        meta_batch_size=2,
        min_weights=[0.1, 0.1],
        max_weights=[5.0, 5.0],
        meta_optimizer='adam'
    )
    return optimizer

# ============================================================================
# Model Fixtures (For test_models.py)
# ============================================================================

@pytest.fixture
def am_setup(mocker):
    """Fixture for AttentionModel"""
    mock_problem = mocker.MagicMock()
    mock_problem.NAME = 'vrpp'
    mock_problem.get_costs.return_value = (torch.zeros(1), {}, None)
    
    mock_encoder = mocker.MagicMock()
    
    # Needs to return tensor on call
    # Needs to return tensor on call
    def mock_enc_fwd(x, edges=None, **kwargs):
        """Mock encoder forward pass."""
        batch, n, dim = x.size()
        return torch.zeros(batch, n, 128) # hidden_dim
    mock_encoder.side_effect = mock_enc_fwd

    # Mock Factory
    class MockFactory(NeuralComponentFactory):
        """Mock factory for neural components."""
        def create_encoder(self, **kwargs):
            """Create mock encoder."""
            return mock_encoder
        def create_decoder(self, **kwargs):
            """Create mock decoder."""
            # Return a MagicMock that acts like AttentionDecoder? 
            # Or real one? Tests verify forward flow. 
            # If we return a Mock, model.decoder becomes that Mock.
            # Then in tests we can configure it.
            m_dec = mocker.MagicMock(spec=AttentionDecoder)
            m_dec.forward.side_effect = lambda input, embeddings, *args, **kwargs: (torch.zeros(1), torch.zeros(1))
            return m_dec

    model = AttentionModel(
        embedding_dim=128,
        hidden_dim=128,
        problem=mock_problem,
        component_factory=MockFactory(),
        n_encode_layers=1,
        n_heads=8,
        checkpoint_encoder=False
    )
    return model

@pytest.fixture
def gat_lstm_setup():
    """Fixture for GATLSTManager"""
    manager = GATLSTManager(
        input_dim_static=2,
        input_dim_dynamic=10,
        hidden_dim=32,
        lstm_hidden=16,
        num_layers_gat=1,
        num_heads=4,
        dropout=0.1,
        device='cpu'
    )
    return manager

@pytest.fixture
def tam_setup(mocker):
    """Fixture for TemporalAttentionModel"""
    mock_problem = mocker.MagicMock()
    mock_problem.NAME = 'vrpp' # To trigger temporal features
    mock_problem.get_costs.return_value = (torch.zeros(1), {}, None)
    
    mock_encoder = mocker.MagicMock()
    def mock_enc_fwd(x, edges=None, **kwargs):
        """Mock encoder forward pass."""
        batch, n, dim = x.size()
        return torch.zeros(batch, n, 128)
    mock_encoder.side_effect = mock_enc_fwd
    
    # We also need to mock GatedRecurrentFillPredictor inside init or just let it exist.
    # It imports from .modules, so we might need real imports for that or mock modules.
    
    # Assuming standard imports work.
    
    class MockActivationFunction(torch.nn.Module):
        """Mock activation function."""
        def __init__(self, *args, **kwargs):
            """Initialize mock activation function."""
            super().__init__()
        def forward(self, x):
            """Mock forward pass."""
            return x

    mocker.patch('logic.src.models.modules.ActivationFunction', new=MockActivationFunction)
    
    # Patch both to be safe
    mocker.patch('logic.src.models.subnets.GatedRecurrentFillPredictor', autospec=False)
    mock_grfp_cls = mocker.patch('logic.src.models.GatedRecurrentFillPredictor', autospec=False)
    
    mock_grfp = mock_grfp_cls.return_value
    def mock_grfp_fwd(x, h=None):
        """Mock GRFP forward pass."""
        return torch.zeros(x.shape[0], 1)

    mock_grfp.side_effect = mock_grfp_fwd
    
    # Mock Factory for TAM
    # Mock Factory for TAM
    class MockTAMFactory(NeuralComponentFactory):
        """Mock factory for TAM components."""
        def create_encoder(self, **kwargs):
            """Create mock encoder."""
            return mock_encoder
        def create_decoder(self, **kwargs):
             """Create mock decoder."""
             m_dec = mocker.MagicMock(spec=AttentionDecoder)
             return m_dec

    model = TemporalAttentionModel(
        embedding_dim=128,
        hidden_dim=128,
        problem=mock_problem,
        component_factory=MockTAMFactory(),
        n_encode_layers=1,
        n_heads=8,
        temporal_horizon=5
    )
    return model


# ============================================================================
# Module Test Fixtures
# ============================================================================

@pytest.fixture
def mock_node_features():
    """Returns a random batch of node features for module testing."""
    batch_size = 2
    num_nodes = 5
    hidden_dim = 16
    return torch.randn(batch_size, num_nodes, hidden_dim)

@pytest.fixture
def mock_adj_matrix():
    """Returns a random adjacency matrix for module testing."""
    batch_size = 2
    num_nodes = 5
    return torch.ones(batch_size, num_nodes, num_nodes)


# ============================================================================
# Session Cleanup
# ============================================================================

def pytest_sessionfinish(session, exitstatus):
    """
    Cleanup artifacts that might be left over after tests, 
    specifically 'test_dehb_output' directory and 'dummy.log'.
    """
    artifacts_to_clean = [
        Path("test_dehb_output"),
        Path("dummy.log")
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

@pytest.fixture
def mock_ppo_deps(mocker):
    """
    Fixture providing mocks specifically for PPO training tests.
    Returns a dictionary containing mock objects for trainer initialization.
    """
    # Mock Model with PPO-specific return values
    mock_model = MagicMock()
    # forward returns: cost, ll, cost_dict, pi, entropy
    mock_model.return_value = (
        torch.tensor([1.0, 1.0]), # cost
        torch.tensor([-1.0, -1.0], requires_grad=True), # ll
        {}, # cost_dict
        torch.tensor([[0, 1], [0, 1]]), # pi
        torch.tensor([0.5, 0.5]) # entropy
    )
    mock_model.to = lambda x: mock_model

    # Mock Optimizer and Baseline
    mock_optimizer = MagicMock()
    mock_baseline = MagicMock()
    # Baseline mocks
    mock_baseline.wrap_dataset = lambda x: x
    mock_baseline.unwrap_batch = lambda x: (x, None)
    mock_baseline.eval.return_value = torch.tensor([1.0, 1.0])

    # Mock Problem
    mock_problem = MagicMock()
    mock_problem.NAME = 'cwcvrp'
    mock_problem.get_costs.return_value = (torch.tensor([1.0, 1.0]), {}, None)

    # Mock Dataset
    dataset_list = [
        {'loc': torch.rand(10, 2), 'demand': torch.rand(10),
         'hrl_mask': torch.zeros(10, 10), 'full_mask': torch.zeros(10, 11)}
        for _ in range(4)
    ]
    mock_dataset = MagicMock()
    mock_dataset.__getitem__ = lambda self, idx: dataset_list[idx]
    mock_dataset.__len__ = lambda self: len(dataset_list)
    mock_dataset.dist_matrix = None
    mock_dataset.has_dist = False

    return {
        'model': mock_model,
        'optimizer': mock_optimizer,
        'baseline': mock_baseline,
        'problem': mock_problem,
        'training_dataset': mock_dataset,
        'val_dataset': []
    }


@pytest.fixture
def mock_dr_grpo_deps():
    """
    Fixture providing mocks specifically for DR-GRPO training tests.
    Returns a dictionary containing mock objects for trainer initialization.
    """
    model = MagicMock()

    # Dynamic return value for model dependent on input size
    # Dynamic return value for model dependent on input size
    def model_side_effect(input, return_pi=False, expert_pi=None, imitation_mode=False, **kwargs):
        """Mock model side effect."""
        if isinstance(input, dict):
            first_val = next(iter(input.values()))
            current_batch_size = first_val.size(0)
        else:
            current_batch_size = 4

        # Returns: cost, log_probs, cost_dict, pi, entropy
        # pi shape: [Batch, SeqLen]
        seq_len = 5
        return (
            torch.randn(current_batch_size, requires_grad=True), # cost
            torch.randn(current_batch_size, requires_grad=True), # log_probs
            {'total': torch.randn(current_batch_size)},          # cost_dict
            torch.randn(current_batch_size, seq_len),            # pi
            torch.randn(current_batch_size, requires_grad=True)  # entropy
        )

    model.side_effect = model_side_effect
    model.to = MagicMock(return_value=model)
    model.train = MagicMock()
    model.eval = MagicMock()
    model.__call__ = MagicMock(side_effect=model_side_effect)

    # Allow setting attributes
    model.decode_type = 'sampling'
    model.set_decode_type = MagicMock()

    optimizer = MagicMock()
    baseline = MagicMock()
    baseline.wrap_dataset.side_effect = lambda x: x
    baseline.unwrap_batch.side_effect = lambda x: (x, None)

    def baseline_eval_side_effect(input, c=None):
         """Mock baseline eval side effect."""
         if isinstance(input, dict):
             first = next(iter(input.values()))
             bs = first.size(0)
         else:
             bs = 4
         return (torch.zeros(bs), torch.zeros(1))

    baseline.eval.side_effect = baseline_eval_side_effect

    dataset = MagicMock()
    dataset.__len__.return_value = 4
    # Return a tensor that can be repeated
    dataset.__getitem__ = MagicMock(return_value={'input': torch.tensor([1.0, 2.0])})

    problem = MagicMock()
    problem.NAME = 'vrpp'

    return {
        'model': model,
        'optimizer': optimizer,
        'baseline': baseline,
        'training_dataset': dataset,
        'val_dataset': dataset,
        'problem': problem
    }


@pytest.fixture
def mock_gspo_deps():
    """
    Fixture providing mocks specifically for GSPO training tests.
    Returns a dictionary containing mock objects for trainer initialization.
    """
    # Mock dependencies
    model = MagicMock()

    # Dynamic return value for model dependent on input size
    # Dynamic return value for model dependent on input size
    def model_side_effect(input, *args, **kwargs):
        """Mock model side effect."""
        if isinstance(input, dict):
            first_val = next(iter(input.values()))
            current_batch_size = first_val.size(0)
        else:
            current_batch_size = 4

        # Returns: cost, log_probs, cost_dict, pi, entropy
        # pi shape: [Batch, SeqLen]
        seq_len = 5
        return (
            torch.randn(current_batch_size, requires_grad=True), # cost
            torch.randn(current_batch_size, requires_grad=True), # log_probs
            {'total': torch.randn(current_batch_size)},          # cost_dict
            torch.randn(current_batch_size, seq_len),            # pi
            torch.randn(current_batch_size, requires_grad=True)  # entropy
        )

    model.side_effect = model_side_effect
    model.to = MagicMock(return_value=model)
    model.train = MagicMock()
    model.eval = MagicMock()

    optimizer = MagicMock()
    baseline = MagicMock()
    baseline.wrap_dataset.side_effect = lambda x: x
    baseline.unwrap_batch.side_effect = lambda x: (x, None)
    baseline.eval.return_value = (torch.zeros(4), torch.zeros(1))

    dataset = MagicMock()
    dataset.__len__.return_value = 4
    dataset.__getitem__ = MagicMock(return_value={'input': torch.tensor([1])})

    problem = MagicMock()
    problem.NAME = 'vrpp'

    return {
        'model': model,
        'optimizer': optimizer,
        'baseline': baseline,
        'training_dataset': dataset,
        'val_dataset': dataset,
        'problem': problem
    }


@pytest.fixture
def mock_sapo_deps():
    """
    Fixture providing mocks specifically for SAPO training tests.
    Returns a dictionary containing mock objects for trainer initialization.
    """
    # Mock dependencies
    model = MagicMock()
    # Model returns: cost, log_probs, cost_dict, pi, entropy
    # Shapes: cost (B), log_probs (B), cost_dict (dict), pi (B, Seq), entropy (B)
    msg_len = 5
    batch_size = 4

    def model_side_effect(input, *args, **kwargs):
        """Mock model side effect."""
        # Check input batch size. Input is dict, e.g. input['depot']
        if isinstance(input, dict):
            first_val = next(iter(input.values()))
            current_batch_size = first_val.size(0)
        else:
            current_batch_size = 4

        return (
            torch.randn(current_batch_size, requires_grad=True), # cost
            torch.randn(current_batch_size, requires_grad=True), # log_probs
            {'total': torch.randn(current_batch_size)},          # cost_dict
            torch.randn(current_batch_size, msg_len),            # pi
            torch.randn(current_batch_size, requires_grad=True)  # entropy
        )

    model.side_effect = model_side_effect
    # Ensure model is callable
    model.to = MagicMock(return_value=model)
    model.train = MagicMock()
    model.eval = MagicMock()

    optimizer = MagicMock()
    baseline = MagicMock()
    # baseline.wrap_dataset returns the dataset itself (identity)
    baseline.wrap_dataset.side_effect = lambda x: x
    baseline.unwrap_batch.side_effect = lambda x: (x, None)
    baseline.eval.return_value = (torch.zeros(4), torch.zeros(1))

    dataset = MagicMock()
    dataset.__len__.return_value = 4
    # Mock dataset iteration
    dataset.__getitem__ = MagicMock(return_value={'input': torch.tensor([1])})

    problem = MagicMock()
    problem.NAME = 'vrpp'

    return {
        'model': model,
        'optimizer': optimizer,
        'baseline': baseline,
        'training_dataset': dataset,
        'val_dataset': dataset,
        'problem': problem
    }
