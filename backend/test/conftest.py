"""
Shared pytest fixtures and configuration for all test modules.

This file is automatically loaded by pytest and provides fixtures
that can be used across all test files.
"""
import os
import sys
import torch
import pytest
import tempfile
import numpy as np
import pandas as pd

from multiprocessing import Lock, Value
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import MagicMock 
from backend.src.pipeline.simulator.bins import Bins
from backend.src.pipeline.simulator import simulation
from backend.src.pipeline.simulator.checkpoints import SimulationCheckpoint

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


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
        'problem': 'tsp',
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
        'area': 'test_area',
        'waste_type': 'paper',
        'policies': ['test_policy_gamma1'],
        'output_dir': 'test_output',
        'checkpoint_dir': 'temp_checkpoints',
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
        'problem': 'cvrp',
    }

# ============================================================================
# Basic Class Fixtures
# ============================================================================

@pytest.fixture
def basic_bins(tmp_path):
    """Returns a basic Bins instance for testing."""
    return Bins(n=10, data_dir=str(tmp_path), sample_dist="gamma")


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
        'backend.src.pipeline.simulator.checkpoints.ROOT_DIR', 
        new=str(mock_root)
    )

    # 2. Define the path structure passed to the constructor
    # The test implies the output_dir input ends with "results"
    mock_output_dir_base = tmp_path / "test_assets" / "results"
    
    # 3. Mock os.listdir since it's used by find_last_checkpoint_day() (called by get_checkpoint_file without 'day')
    mocker.patch('os.listdir', return_value=[])

    # 4. Initialize SimulationCheckpoint
    from backend.src.pipeline.simulator.checkpoints import SimulationCheckpoint
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
    mocker.patch('backend.src.data.generate_data.generate_tsp_data', return_value=[(None, None)])
    mocker.patch('backend.src.data.generate_data.generate_vrp_data', return_value=[(None, None)])
    mocker.patch('backend.src.data.generate_data.generate_pctsp_data', return_value=[(None, None)])
    mocker.patch('backend.src.data.generate_data.generate_op_data', return_value=[(None, None)])
    mocker.patch('backend.src.data.generate_data.generate_vrpp_data', return_value=[(None, None)])
    mocker.patch('backend.src.data.generate_data.generate_wcrp_data', return_value=[(None, None)])
    mocker.patch('backend.src.data.generate_data.generate_wcvrp_data', return_value=[(None, None)])
    mocker.patch('backend.src.data.generate_data.generate_pdp_data', return_value=[(None, None)])
    
    # 2. Mock the WSR simulator data generator
    mocker.patch('backend.src.data.generate_data.generate_wsr_data', return_value=[(None, None)])
    
    # 3. Mock the save utility
    mocker.patch('backend.src.data.generate_data.save_dataset', return_value=None)


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
    mock_bins.get_fill_history.return_value = np.array([[10, 20], [30, 40]])
    return mock_bins


@pytest.fixture
def mock_sim_dependencies(mocker, tmp_path, mock_bins_instance):
    """
    Mocks all major external dependencies for single_simulation
    and sequential_simulations.
    """
    # 1. Patch ROOT_DIR
    mocker.patch('backend.src.pipeline.simulator.simulation.ROOT_DIR', str(tmp_path))
    
    # 2. Mock loader functions
    mock_depot = pd.DataFrame({'ID': [0], 'Lat': [40], 'Lng': [-8]})
    mock_data = pd.DataFrame({'ID': [1, 2], 'Stock': [10, 20]})
    mock_coords = pd.DataFrame({'ID': [1, 2], 'Lat': [40.1, 40.2], 'Lng': [-8.1, -8.2]})
    mocker.patch('backend.src.pipeline.simulator.simulation.load_depot', return_value=mock_depot)
    mocker.patch('backend.src.pipeline.simulator.simulation.load_simulator_data', 
                 return_value=(mock_data.copy(), mock_coords.copy()))

    # 3. Mock processor functions
    mock_proc_data = pd.DataFrame({'ID': [0, 1, 2], 'Stock': [0, 10, 20]})
    mock_proc_coords = pd.DataFrame({'ID': [0, 1, 2], 'Lat': [40, 40.1, 40.2], 'Lng': [-8, -8.1, -8.2]})
    mocker.patch('backend.src.pipeline.simulator.simulation.process_data', 
                 return_value=(mock_proc_data.copy(), mock_proc_coords.copy()))
    mocker.patch('backend.src.pipeline.simulator.simulation.process_model_data', 
                 return_value=('mock_model_tup_0', 'mock_model_tup_1'))

    # 4. Mock network functions
    mock_dist_tup = (np.array([[0, 1], [1, 0]]), 'mock_paths', 'mock_tensor', 'mock_distC')
    mock_adj_matrix = np.array([[1, 1], [1, 1]])
    mocker.patch('backend.src.pipeline.simulator.simulation._setup_dist_path_tup', 
                 return_value=(mock_dist_tup, mock_adj_matrix))
    mocker.patch('backend.src.pipeline.simulator.simulation.compute_distance_matrix', 
                 return_value=np.array([[0,1],[1,0]]))
    mocker.patch('backend.src.pipeline.simulator.simulation.apply_edges', 
                 return_value=('mock_dist_edges', 'mock_paths', 'mock_adj'))
    mocker.patch('backend.src.pipeline.simulator.simulation.get_paths_between_states', 
                 return_value='mock_all_paths')

    # 5. Mock Bins class
    mocker.patch('backend.src.pipeline.simulator.simulation.Bins', return_value=mock_bins_instance)

    # 6. Mock setup functions
    mocker.patch('backend.src.pipeline.simulator.simulation.setup_model', 
                 return_value=('mock_model_env', 'mock_configs'))
    mocker.patch('backend.src.pipeline.simulator.simulation.setup_env', 
                 return_value='mock_or_env')

    # 7. Mock day function
    mock_dlog = {'day': 1, 'overflows': 0, 'kg_lost': 0, 'kg': 0, 'ncol': 0, 'km': 0, 'kg/km': 0, 'tour': [0]}
    mock_data_ls = (mock_proc_data, mock_proc_coords, mock_bins_instance)
    mock_output_ls = (0, mock_dlog, {}) # overflows, dlog, output_dict
    mocker.patch('backend.src.pipeline.simulator.simulation.run_day', 
                 return_value=(mock_data_ls, mock_output_ls, None)) # data_ls, output_ls, cached

    # 8. Mock checkpointing
    mock_cp_instance = mocker.MagicMock()
    mock_cp_instance.load_state.return_value = (None, 0) # Default: no resume
    mocker.patch('backend.src.pipeline.simulator.simulation.SimulationCheckpoint', 
                 return_value=mock_cp_instance)
    
    # Mock the context manager and its hook
    mock_hook = mocker.MagicMock()
    mock_hook.day = 0
    mocker.patch('backend.src.pipeline.simulator.simulation.checkpoint_manager', 
                 return_value=nullcontext(mock_hook)) # Use nullcontext to just yield the hook
    mocker.patch('backend.src.pipeline.simulator.simulation.CheckpointHook', 
                 return_value=mock_hook)

    # 9. Mock utilities
    mocker.patch('backend.src.pipeline.simulator.simulation.log_to_json')
    mocker.patch('backend.src.pipeline.simulator.simulation.output_stats',
                 return_value=({}, {})) # mock_log, mock_log_std
    mocker.patch('backend.src.pipeline.simulator.simulation.save_matrix_to_excel')
    mocker.patch('backend.src.pipeline.simulator.simulation.tqdm', side_effect=lambda x, **kwargs: x)
    mocker.patch('backend.src.pipeline.simulator.simulation.time.process_time', return_value=1.0)
    mocker.patch('os.makedirs', new_callable=lambda: lambda *args, **kwargs: None)
    mocker.patch('pandas.DataFrame.to_excel')
    mocker.patch('statistics.mean', return_value=1.0)
    mocker.patch('statistics.stdev', return_value=0.1)

    # Return key mocks for modification in tests
    return {
        'checkpoint': mock_cp_instance,
        'hook': mock_hook,
        'run_day': simulation.run_day,
        'log_to_json': simulation.log_to_json,
        'save_excel': simulation.save_matrix_to_excel,
        'bins': mock_bins_instance,
        'setup_model': simulation.setup_model,
        'setup_env': simulation.setup_env,
        'process_data': simulation.process_data,
        '_setup_dist_path_tup': simulation._setup_dist_path_tup,
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
    # We must mock it where it's *used*, e.g., 'backend.src.utils.definitions'
    mock_udef = MagicMock()
    mock_udef.WASTE_TYPES = {
        'paper': 'Embalagens de papel e cartão',
        'plastic': 'Mistura de embalagens'
    }
    mock_udef.LOCK_TIMEOUT = 30
    mocker.patch('backend.src.utils.definitions', mock_udef)
    
    # Return the mocks so tests can configure them
    return mock_read_csv, mock_read_excel, mock_udef


@pytest.fixture
def mock_run_day_deps(mocker):
    """
    Mocks all dependencies for the run_day function, including policy 
    and GUI logging mocks injected as part of the fixture.
    """
    # 1. Mock the 'bins' object (Keep existing logic)
    mock_bins = MagicMock()
    mock_bins.is_stochastic.return_value = False 
    mock_bins.loadFilling.return_value = (0, 'mock_fill', 'mock_sum_lost')
    mock_bins.stochasticFilling.return_value = (0, 'mock_stoch_fill', 'mock_stoch_sum_lost')
    mock_bins.c = np.array([10, 20])
    mock_bins.n = 2
    mock_bins.collect.return_value = (100.0, 2) # Added mock for collect

    # 2. Mock other required arguments (Keep existing logic)
    mock_new_data = pd.DataFrame() 
    mock_coords = pd.DataFrame()
    mock_model_env = MagicMock()
    mock_model_ls = (MagicMock(), MagicMock(), MagicMock())
    mock_dist_tup = (
        MagicMock(), # distance_matrix
        MagicMock(), # paths_between_states
        MagicMock(), # dm_tensor
        MagicMock()  # distancesC 
    )

    # 3. Patch the functions that are actually called (CRUCIAL CHANGE)
    
    # Mock policy_regular to return a tour (to prevent errors in get_daily_results)
    mock_policy_regular = mocker.patch(
        'backend.src.or_policies.regular.policy_regular', return_value=[0, 1, 2, 0] 
    )
    # Mock send_daily_output_to_gui to prevent the final crash
    mock_send_output = mocker.patch(
        'backend.src.pipeline.simulator.day.send_daily_output_to_gui', autospec=True
    )
    # Mock get_route_cost (used by policy_regular flow)
    mocker.patch('backend.src.or_policies.single_vehicle.get_route_cost', return_value=50.0)

    # 4. Return the dependencies, including the new mocks
    return {
        'bins': mock_bins,
        'model_env': mock_model_env,
        'model_ls': mock_model_ls,
        'distpath_tup': mock_dist_tup,
        'new_data': mock_new_data,
        'coords': mock_coords,
        # INJECTING the mock objects into the dictionary:
        'mock_policy_regular': mock_policy_regular,
        'mock_send_output': mock_send_output,
    }

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
