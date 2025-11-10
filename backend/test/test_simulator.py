import os
import json
import torch
import pytest
import statistics
import numpy as np
import pandas as pd

from multiprocessing import Lock
from unittest.mock import MagicMock
from pandas.testing import assert_frame_equal
from ..src.pipeline.simulator.bins import Bins
from ..src.pipeline.simulator import simulation
from ..src.pipeline.simulator.network import compute_distance_matrix, apply_edges
from ..src.pipeline.simulator.day import set_daily_waste, get_daily_results, run_day
from ..src.pipeline.simulator.checkpoints import checkpoint_manager, CheckpointError
from ..src.pipeline.simulator.loader import (
    load_simulator_data,
    load_indices, load_depot, 
    load_area_and_waste_type_params
)
from ..src.pipeline.simulator.processor import (
    process_data, process_coordinates,
    sort_dataframe, setup_df, haversine_distance,
)


class TestBins:
    """Tests for the WSmart+ Route simulator bins."""
    @pytest.mark.unit
    def test_bins_init(self, tmp_path):
        """Test the initialization of the Bins class."""
        bins = Bins(n=5, data_dir=str(tmp_path), sample_dist="gamma")
        assert bins.n == 5
        assert bins.distribution == "gamma"
        assert np.all(bins.c == 0)
        assert np.all(bins.means == 10)
        assert len(bins.indices) == 5

    @pytest.mark.unit
    def test_bins_init_emp(self, mocker, tmp_path):
        """Test initialization with 'emp' distribution, mocking the grid."""
        mocker.patch('backend.src.pipeline.simulator.bins.OldGridBase', autospec=True)
        bins = Bins(n=5, data_dir=str(tmp_path), sample_dist="emp", area="test_area")
        assert bins.distribution == "emp"
        assert bins.grid is not None

    @pytest.mark.unit
    def test_bins_init_invalid_dist(self, tmp_path):
        """Test that initialization fails with an invalid distribution."""
        with pytest.raises(AssertionError):
            Bins(n=5, data_dir=str(tmp_path), sample_dist="invalid_dist")

    @pytest.mark.unit
    def test_bins_collect(self, basic_bins):
        """Test the collect method."""
        basic_bins.c = np.array([10, 80, 90, 0, 50, 0, 0, 0, 0, 0], dtype=float)
        basic_bins.ncollections = np.zeros((10))
        
        tour = [0, 1, 2, 0] # Collect from bins 1 and 2
        collected_kg, num_collections = basic_bins.collect(tour)
        
        assert collected_kg == 90  # 10 + 80
        assert num_collections == 2
        assert basic_bins.c[0] == 0  # Bin 1 collected
        assert basic_bins.c[1] == 0  # Bin 2 collected
        assert basic_bins.c[4] == 50 # Bin 5 unchanged
        assert basic_bins.ncollections[0] == 1
        assert basic_bins.ncollections[1] == 1
        assert basic_bins.ncollections[4] == 0

    @pytest.mark.unit
    def test_bins_collect_empty_tour(self, basic_bins):
        """Test collect method with an empty or depot-only tour."""
        basic_bins.c = np.ones((10)) * 50
        
        collected_kg, num_collections = basic_bins.collect([0])
        assert collected_kg == 0
        assert num_collections == 0
        
        collected_kg, num_collections = basic_bins.collect([0, 0])
        assert collected_kg == 0
        assert num_collections == 0
        assert np.all(basic_bins.c == 50) # No change

    @pytest.mark.unit
    def test_bins_stochastic_filling_gamma(self, mocker, basic_bins):
        """Test stochastic filling with gamma distribution."""
        basic_bins.c = np.ones((10)) * 90.0
        basic_bins.lost = np.zeros((10))
        
        # Mock the gamma random variable sampler to return a fixed value (e.g., 20)
        mock_rvs = mocker.patch('numpy.random.gamma', return_value=np.ones((1, 10)) * 20.0)
        
        new_overflows, fill, sum_lost = basic_bins.stochasticFilling()
        
        assert mock_rvs.called
        assert new_overflows == 10 # All 10 bins overflowed
        assert sum_lost == 100.0 # 10 lost from each of the 10 bins
        assert np.all(basic_bins.c == 100.0) # All bins are full
        assert np.all(basic_bins.lost == 10.0) # 10kg lost from each bin
        assert len(basic_bins.history) == 1
        assert basic_bins.ndays == 1

    @pytest.mark.unit
    def test_bins_set_gamma_distribution(self, basic_bins):
        """Test setting gamma distribution parameters."""
        basic_bins.setGammaDistribution(option=1)
        assert basic_bins.distribution == 'gamma'
        # Check if params are set based on the logic in option 1
        # This assumes knowledge of the implementation
        assert basic_bins.dist_param1[0] in [2, 6] 
        assert basic_bins.dist_param2[0] in [6, 4]


class TestCheckpoints:
    """Tests for the WSmart+ Route simulator checkpoint manager."""
    @pytest.mark.unit
    def test_checkpoint_init(self, basic_checkpoint):
        """Test SimulationCheckpoint initialization."""
        assert basic_checkpoint.policy == "test_policy"
        assert basic_checkpoint.sample_id == 1
        assert str(basic_checkpoint.checkpoint_dir).endswith("temp")
        assert str(basic_checkpoint.output_dir).endswith("results/temp")

    @pytest.mark.unit
    def test_get_checkpoint_file(self, basic_checkpoint):
        """Test the generation of checkpoint filenames."""
        fname = basic_checkpoint.get_checkpoint_file(day=5)
        assert fname.endswith("temp/checkpoint_test_policy_1_day5.pkl")
        
        fname_end = basic_checkpoint.get_checkpoint_file(day=10, end_simulation=True)
        assert fname_end.endswith("results/temp/checkpoint_test_policy_1_day10.pkl")

    @pytest.mark.unit
    def test_save_load_state(self, basic_checkpoint, tmp_path):
        """Test saving and loading a simulation state."""
        state = {'day': 10, 'bins': np.array([1, 2, 3])}
        
        # Save state
        basic_checkpoint.save_state(state, day=10)
        expected_file = basic_checkpoint.get_checkpoint_file(day=10)
        assert os.path.exists(expected_file)
        
        # Load state
        loaded_state, last_day = basic_checkpoint.load_state(day=10)
        assert last_day == 10
        assert loaded_state['day'] == 10
        assert np.array_equal(loaded_state['bins'], state['bins'])

    @pytest.mark.unit
    def test_load_latest_state(self, basic_checkpoint, mocker):
        """Test loading the latest available checkpoint."""
        state_5 = {'day': 5}
        state_10 = {'day': 10}
        
        # These calls write the files to the filesystem
        basic_checkpoint.save_state(state_5, day=5)
        basic_checkpoint.save_state(state_10, day=10)
        
        # --- FIX: Mock os.listdir to reflect the files saved ---
        # This allows find_last_checkpoint_day() to "see" the files.
        saved_filenames = [
            "checkpoint_test_policy_1_day5.pkl",
            "checkpoint_test_policy_1_day10.pkl",
            "some_ignored_file.txt" # Add noise to ensure filtering works
        ]
        # Patch the os module where it is imported in checkpoints.py
        mocker.patch('backend.src.pipeline.simulator.checkpoints.os.listdir', return_value=saved_filenames)
        # --------------------------------------------------------

        # find_last_checkpoint_day should find day 10
        assert basic_checkpoint.find_last_checkpoint_day() == 10
        
        # load_state (with day=None) should load day 10
        loaded_state, last_day = basic_checkpoint.load_state()
        assert last_day == 10
        assert loaded_state['day'] == 10

    @pytest.mark.unit
    def test_checkpoint_manager_success(self, mocker, basic_checkpoint):
        """Test the checkpoint_manager context on successful completion."""
        mocker.patch.object(basic_checkpoint, 'save_state', autospec=True)
        mocker.patch.object(basic_checkpoint, 'clear', autospec=True)
        
        state = {'data': 'final_state'}
        state_getter = lambda: state
        
        with checkpoint_manager(basic_checkpoint, checkpoint_interval=5, state_getter=state_getter) as hook:
            hook.day = 10
            hook.after_day(tic=1.0) # Simulates end of day 10
        
        # on_completion should be called
        basic_checkpoint.save_state.assert_called_with(state, 10, end_simulation=True)
        basic_checkpoint.clear.assert_called_with(None, None)

    @pytest.mark.unit
    def test_checkpoint_manager_error(self, mocker, basic_checkpoint):
        """Test the checkpoint_manager context on simulation error."""
        mocker.patch.object(basic_checkpoint, 'save_state', autospec=True)
        
        state = {'data': 'error_state'}
        state_getter = lambda: state
        error_to_raise = ValueError("Simulation Failed")
        
        with pytest.raises(CheckpointError) as e:
            with checkpoint_manager(basic_checkpoint, checkpoint_interval=5, state_getter=state_getter) as hook:
                # Set the day to ensure it's logged correctly
                hook.day = 7 
                raise error_to_raise
                
        basic_checkpoint.save_state.assert_called_with(state, 7)
        
        assert e.value.error_result['error'] == "Simulation Failed"
        assert e.value.error_result['day'] == 7
    

class TestLoader:
    """Tests for the WSmart+ Route simulator data loader."""
    @pytest.mark.unit
    def test_load_area_and_waste_type_params(self):
        """Test loading parameters for a known area and waste type."""
        vehicle_capacity, revenue, density, expenses, bin_volume = load_area_and_waste_type_params(
            area='Rio Maior', waste_type='paper'
        )
        assert vehicle_capacity == 4000
        assert density == 21.0

        vehicle_capacity, revenue, density, expenses, bin_volume = load_area_and_waste_type_params(
            area='Figueira da Foz', waste_type='plastic'
        )
        assert vehicle_capacity == 2500
        assert density == 20.0

    @pytest.mark.unit
    def test_load_area_waste_type_invalid(self):
        """Test assertions for unknown area or waste type."""
        with pytest.raises(AssertionError):
            load_area_and_waste_type_params(area='Unknown Area', waste_type='paper')
        
        with pytest.raises(AssertionError):
            load_area_and_waste_type_params(area='Rio Maior', waste_type='unknown_waste')

    @pytest.mark.unit
    def test_load_depot(self, mock_data_dir, mocker):
        """Test loading depot coordinates from the mock Facilities.csv."""
        # mock_data_dir setup creates Facilities.csv with 'RM' for 'riomaior'
        # udef.MAP_DEPOTS['riomaior'] is assumed to be 'RM'
        mocker.patch(
            'backend.src.pipeline.simulator.loader.udef.MAP_DEPOTS',
            {'riomaior': 'RM'}
        )

        depot_df = load_depot(data_dir=mock_data_dir, area='Rio Maior')
        
        assert len(depot_df) == 1
        assert depot_df.loc[0, 'ID'] == 0
        assert depot_df.loc[0, 'Lat'] == 40.0
        assert depot_df.loc[0, 'Lng'] == -8.0
        assert depot_df.loc[0, 'Stock'] == 0

    @pytest.mark.unit
    def test_load_indices_file_exists(self, mocker, tmp_path):
        """Test loading indices when the JSON file already exists."""
        indices_dir = tmp_path / "data" / "wsr_simulator" / "bins_selection"
        indices_dir.mkdir(parents=True)
        indices_file = indices_dir / "test_indices.json"
        
        mock_indices = [[1, 2, 3], [4, 5, 6]]
        with open(indices_file, 'w') as f:
            json.dump(mock_indices, f)
        
        # Mock udef.ROOT_DIR to point to tmp_path
        mocker.patch('backend.src.utils.definitions.ROOT_DIR', str(tmp_path))
        
        indices = load_indices("test_indices.json", n_samples=2, n_nodes=3, data_size=100)
        assert indices == mock_indices

    @pytest.mark.unit
    def test_load_indices_create_file(self, mocker, tmp_path):
        """Test creating indices when the file does not exist."""
        indices_dir = tmp_path / "data" / "wsr_simulator" / "bins_selection"
        indices_dir.mkdir(parents=True)
        indices_file = indices_dir / "new_indices.json"
        
        # Mock udef.ROOT_DIR
        mocker.patch('backend.src.utils.definitions.ROOT_DIR', str(tmp_path))
        # Mock pd.Series.sample to be deterministic
        mocker.patch('pandas.Series.sample', side_effect=[
            pd.Series([10, 20]), pd.Series([30, 40])
        ])

        indices = load_indices("new_indices.json", n_samples=2, n_nodes=2, data_size=100)
        
        assert len(indices) == 2
        assert indices[0] == [10, 20] # Note: the function sorts the list
        assert indices[1] == [30, 40]
        assert indices_file.exists() # File should be created
        with open(indices_file, 'r') as f:
            data = json.load(f)
        assert data == [[10, 20], [30, 40]]

    @pytest.mark.unit
    def test_load_simulator_data_riomaior_preprocessing(self, mock_load_dependencies):
        """
        Tests the 'riomaior' path and its internal _preprocess functions.
        """
        mock_read_csv, _, _ = mock_load_dependencies

        # --- Arrange ---
        mock_rate_data = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            '101': [10.1, 20.9, 30.0],  # Stock=10, Mean=20.3 (fillna(0))
            '102': [-5, 2.2, 5.0],     # Stock=2, Mean=2.4 (fillna(0))
            '103': [1.0, 1.0, -1.0],   # Stock=1, Mean=0.67 (fillna(0))
        })
        mock_info_data = pd.DataFrame({
            'ID': [102, 101, 104], # Note: 103 is missing, 104 is extra
            'Latitude': [40.0, 41.0, 42.0],
            'Longitude': [-8.0, -9.0, -10.0],
            'Tipo de Residuos': ['Residuos Urbanos', 'Plastico', 'Residuos Urbanos']
        })

        # Configure the mock to return the correct DataFrame
        def csv_side_effect(path):
            if 'crude_rate' in path:
                return mock_rate_data.copy()
            if 'info' in path:
                return mock_info_data.copy()
            return pd.DataFrame()
        mock_read_csv.side_effect = csv_side_effect

        # --- Act ---
        data_df, coords_df = load_simulator_data(
            data_dir='fake/dir',
            number_of_bins=300, # <= 317
            area='Rio Maior',
            waste_type=None,
            lock=None
        )

        # --- Assert ---
        # 1. Check Coords (filters, renames, sorts, resets index)
        expected_coords = pd.DataFrame({
            'ID': [101, 102],
            'Lat': [41.0, 40.0],
            'Lng': [-9.0, -8.0],
        }).sort_values(by='ID').reset_index(drop=True)
        assert_frame_equal(coords_df, expected_coords, check_dtype=False)

        # 2. Check Data (tests preprocessing, normalization, filtering, sorting)
        expected_data = pd.DataFrame({
            'ID': [101, 102],
            'Stock': [1.0, 0.111111],
            'Accum_Rate': [1.0, 0.084746]
        }).sort_values(by='ID').reset_index(drop=True)
        
        assert_frame_equal(data_df, expected_data, check_dtype=False, atol=1e-5)

    @pytest.mark.unit
    def test_load_simulator_data_riomaior_with_waste_type(self, mock_load_dependencies):
        """
        Tests the 'riomaior' path with a 'waste_type' filter.
        """
        mock_read_csv, _, mock_udef = mock_load_dependencies
        
        # --- Arrange ---
        mock_rate_data = pd.DataFrame({
            'Date': ['2023-01-01'], '101': [10.0], '102': [20.0], '105': [30.0]
        })
        mock_info_data = pd.DataFrame({
            'ID': [101, 102, 105],
            'Latitude': [40.0, 41.0, 42.0],
            'Longitude': [-8.0, -9.0, -10.0],
            'Tipo de Residuos': ['Embalagens de papel e cartão', 'Mistura de embalagens', 'Embalagens de papel e cartão']
        })
        for col in ['101', '102', '105']:
            mock_rate_data[col] = mock_rate_data[col].astype(float)
        
        def csv_side_effect(path):
            if 'crude_rate' in path: return mock_rate_data.copy()
            if 'info' in path: return mock_info_data.copy()
        mock_read_csv.side_effect = csv_side_effect
        
        # --- Act ---
        # Ask for 'Papel' waste type (maps to 'Papel')
        data_df, coords_df = load_simulator_data(
            'fake/dir', 300, 'Rio Maior', 'paper', None
        )
        
        # --- Assert ---
        # Should filter out bin 102 ('Plastico')
        expected_coords = pd.DataFrame({
            'ID': [101, 105],
            'Lat': [40.0, 42.0],
            'Lng': [-8.0, -10.0],
        }).sort_values(by='ID').reset_index(drop=True)
        assert_frame_equal(coords_df, expected_coords, check_dtype=False)
        
        # Data for 101 (Stock=10, Rate=10) and 105 (Stock=30, Rate=30)
        # Min=10, Max=30 -> Norm 101=(0.0, 0.0), 105=(1.0, 1.0)
        expected_data = pd.DataFrame({
            'ID': [101, 105],
            'Stock': [0.0, 1.0],
            'Accum_Rate': [0.0, 1.0]
        }).sort_values(by='ID').reset_index(drop=True)
        assert_frame_equal(data_df, expected_data, check_dtype=False)

    @pytest.mark.unit
    @pytest.mark.parametrize("n_bins, expected_file", [
        (20, 'StockAndAccumulationRate - small.xlsx'),
        (50, 'StockAndAccumulationRate - 50bins.xlsx'),
        (100, 'StockAndAccumulationRate.xlsx'),
    ])
    def test_load_simulator_data_mixrmbac_bin_logic(self, mock_load_dependencies, n_bins, expected_file):
        """
        Tests the 'mixrmbac' path bin number logic.
        """
        _, mock_read_excel, _ = mock_load_dependencies

        # --- Arrange ---
        mock_data = pd.DataFrame({'ID': [1, 3, 2], 'Stock': [10, 30, 20]})
        mock_coords = pd.DataFrame({'ID': [2, 1, 3], 'Lat': [1, 2, 3], 'Lng': [1, 2, 3]})
        
        def excel_side_effect(path):
            if 'StockAndAccumulationRate' in path: return mock_data.copy()
            if 'Coordinates' in path: return mock_coords.copy()
        mock_read_excel.side_effect = excel_side_effect

        # --- Act ---
        data_df, coords_df = load_simulator_data(
            data_dir='fake/dir',
            number_of_bins=n_bins,
            area='mixrmbac',
            lock=None
        )

        # --- Assert ---
        # Check that the correct files were called
        data_path = f'fake/dir/bins_waste/{expected_file}'
        coords_path = data_path.replace('bins_waste', 'coordinates').replace('StockAndAccumulationRate', 'Coordinates')
        
        mock_read_excel.assert_any_call(data_path)
        mock_read_excel.assert_any_call(coords_path)
        
        # Check data is sorted and index is reset
        assert_frame_equal(data_df, mock_data.sort_values(by='ID').reset_index(drop=True))
        assert_frame_equal(coords_df, mock_coords.sort_values(by='ID').reset_index(drop=True))

    @pytest.mark.unit
    @pytest.mark.parametrize("area, n_bins, expected_error", [
        ('mixrmbac', 300, "Number of bins for area mixrmbac must be <= 225"),
        ('riomaior', 400, "Number of bins for area riomaior must be <= 317"),
        ('figueiradafoz', 2000, "Number of bins for area figueiradafoz must be <= 1094"),
        ('both', 600, "Number of bins for both must be <= 542"),
        ('invalid_area', 100, "Invalid area: invalidarea"),
    ])
    def test_load_simulator_data_raises_assertion_error(self, mock_load_dependencies, area, n_bins, expected_error):
        """
        Tests that the function correctly raises AssertionErrors for invalid inputs.
        """
        mock_read_csv, mock_read_excel, _ = mock_load_dependencies
        mock_read_csv.return_value = pd.DataFrame({'Date':['2023-01-01'], '1':[1]})
        mock_read_excel.return_value = pd.DataFrame({'ID':[1]})

        with pytest.raises(AssertionError) as exc_info:
            load_simulator_data(
                data_dir='fake/dir',
                number_of_bins=n_bins,
                area=area,
                lock=None
            )
        
        assert str(exc_info.value) == expected_error

    @pytest.mark.unit
    def test_load_simulator_data_with_lock(self, mock_load_dependencies):
        """
        Tests that the lock's acquire() and release() methods are called.
        """
        _, mock_read_excel, mock_udef = mock_load_dependencies
        
        mock_data = pd.DataFrame({'ID': [1], 'Stock': [0.5], 'Accum_Rate': [0.5]})
        mock_coords = pd.DataFrame({'ID': [1], 'Lat': [40.0], 'Lng': [-8.0]})
        mock_read_excel.side_effect = [mock_data, mock_coords] 
        
        # --- Arrange ---
        mock_lock = MagicMock(spec=Lock())
        
        # Set up the context manager methods to return the mock itself
        # and ensure acquire/release are callable.
        mock_lock.acquire.return_value = True
        mock_lock.__enter__.return_value = mock_lock

        # --- Act ---
        load_simulator_data(
            data_dir='fake/dir',
            number_of_bins=10,
            area='mixrmbac',
            lock=mock_lock
        )

        # --- Assert ---     
        # Check that acquire was called (potentially without args)
        mock_lock.acquire.assert_called_once()
        mock_lock.release.assert_called_once()


class TestProcessor:
    """Tests for the WSmart+ Route simulator data processor."""
    @pytest.mark.unit
    def test_sort_dataframe(self):
        """Test sorting a DataFrame by a specific metric."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [30, 20, 10]})
        sorted_df = sort_dataframe(df, 'B', ascending_order=True)
        assert sorted_df['B'].tolist() == [10, 20, 30]
        assert sorted_df.columns[0] == 'B' # Metric should be the first column

    @pytest.mark.unit
    def test_setup_df(self, mock_coords_df, mock_data_df):
        """Test the setup_df function."""
        depot, _ = mock_coords_df
        
        processed_df = setup_df(depot, mock_data_df, col_names=['ID', 'Stock', 'Accum_Rate'])
        
        assert len(processed_df) == 3 # 2 bins + 1 depot
        assert processed_df.loc[0, 'ID'] == 0 # Depot is row 0
        assert processed_df.loc[0, 'Stock'] == 0
        assert processed_df.loc[1, 'ID'] == 1
        assert processed_df.loc[1, 'Stock'] == 10
        assert '#bin' in processed_df.columns

    @pytest.mark.unit
    def test_process_data(self, mock_data_df, mock_coords_df):
        """Test the main data processing function."""
        depot, coords = mock_coords_df
        data = mock_data_df
        
        new_data, new_coords = process_data(data, coords, depot, indices=None)
        
        # Check new_data
        assert len(new_data) == 3
        assert new_data.loc[0, 'ID'] == 0
        assert new_data.loc[1, 'ID'] == 1
        assert 'Stock' in new_data.columns
        
        # Check new_coords
        assert len(new_coords) == 3
        assert new_coords.loc[0, 'ID'] == 0
        assert new_coords.loc[2, 'ID'] == 2
        assert 'Lat' in new_coords.columns

    @pytest.mark.unit
    def test_haversine_distance(self):
        """Test the Haversine distance calculation."""
        # Point to itself
        dist = haversine_distance(40.0, -8.0, 40.0, -8.0)
        assert np.isclose(dist, 0.0)
        
        # Known distance (approx Lisbon to Porto)
        lisbon_lat, lisbon_lon = 38.7223, -9.1393
        porto_lat, porto_lon = 41.1579, -8.6291
        dist = haversine_distance(lisbon_lat, lisbon_lon, porto_lat, porto_lon)
        assert np.isclose(dist, 274, atol=2) # Approx 274 km

    @pytest.mark.unit
    def test_process_coordinates_mmn(self):
        """Test coordinate processing with min-max normalization."""
        coords = pd.DataFrame({
            'Lat': [40.0, 40.1, 40.2],
            'Lng': [-8.0, -8.0, -8.2]
        })
        
        depot, loc = process_coordinates(coords, method='mmn', col_names=['Lat', 'Lng'])
        
        # Depot (index 0): Lng = 1.0, Lat = 0.0 → [1.0, 0.0]
        assert np.allclose(depot, [1.0, 0.0])

        # Bins:
        # Bin1: Lat=(40.1-40.0)/(40.2-40.0)=0.5,   Lng=(-8.0 - -8.2)/(0.2)=1.0
        # Bin2: Lat=1.0,                          Lng=(-8.2 - -8.2)/0.2 = 0.0
        expected_loc = np.array([
            [1.0, 0.5],   # Bin1: [Lng, Lat]
            [0.0, 1.0]    # Bin2
        ])
        assert np.allclose(loc, expected_loc)


class TestNetwork:
    """Tests for the WSmart+ Route simulator network graph computations."""
    @pytest.mark.unit
    def test_compute_distance_matrix_hsd(self):
        """Test 'hsd' (haversine) distance matrix calculation."""
        coords = pd.DataFrame({
            'ID': [0, 1, 2],
            'Lat': [40.0, 40.0, 41.0],
            'Lng': [-8.0, -9.0, -8.0]
        })
        
        dist_matrix = compute_distance_matrix(coords, method='hsd')
        
        assert dist_matrix.shape == (3, 3)
        assert np.all(np.diag(dist_matrix) == 0)
        assert dist_matrix[0, 1] > 0 # (0,0) to (0,1)
        assert dist_matrix[0, 2] > 0 # (0,0) to (1,0)
        assert dist_matrix[1, 2] > 0
        assert np.isclose(dist_matrix[0, 1], dist_matrix[1, 0]) # Should be symmetric

    @pytest.mark.unit
    def test_compute_distance_matrix_load_file(self, mocker, tmp_path):
        """Test loading a distance matrix from a file."""
        mocker.patch('backend.src.utils.definitions.ROOT_DIR', str(tmp_path))
        
        dm_dir = tmp_path / "data" / "wsr_simulator" / "distance_matrix"
        dm_dir.mkdir(parents=True, exist_ok=True)
        dm_file = dm_dir / "test_dm.csv"
        
        # Perfect CSV: header = IDs, first column = IDs
        dm_file.write_text("ID,1,2\n1,0,10\n2,10,0")
        
        coords = pd.DataFrame({'ID': [1, 2], 'Lat': [0.0, 0.0], 'Lng': [0.0, 0.0]})
        
        dist_matrix = compute_distance_matrix(coords, method='hsd', dm_filepath=str(dm_file))
        
        expected = np.array([[0, 10], [10, 0]])
        assert np.array_equal(dist_matrix, expected)

    @pytest.mark.unit
    def test_apply_edges(self):
        """Test the apply_edges function to threshold a distance matrix."""
        dist_matrix = np.array([
            [0, 10, 20],
            [10, 0, 5],
            [20, 5, 0]
        ], dtype=float)
        
        threshold = 0.15
        
        dm_edges, paths, adj_matrix = apply_edges(dist_matrix, threshold, edge_method='knn')
        expected_adj = np.array([
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0]
        ])
        assert np.array_equal(adj_matrix, expected_adj)

        row_idx = paths[(2, 1)][:-1]
        col_idx = paths[(2, 1)][1:]
        assert dm_edges[0, 0] == 0
        assert dm_edges[2, 1] == dm_edges[row_idx, col_idx].sum()
        assert dm_edges[0, 1] == 10
        assert dm_edges[2, 0] == 20


class TestDay:
    """Tests for the WSmart+ Route simulation day."""
    @pytest.mark.unit
    def test_set_daily_waste(self):
        """Test the function that sets daily waste in model_data."""
        model_data = {'fill_history': torch.tensor([])}
        waste = np.array([10, 20, 30])
        fill = np.array([5, 15, 25])
        device = torch.device('cpu')
        
        model_data = set_daily_waste(model_data, waste, device, fill)
        
        assert 'waste' in model_data
        assert 'current_fill' in model_data
        assert torch.is_tensor(model_data['waste'])
        # Check scaling (divided by 100)
        assert torch.allclose(model_data['waste'], torch.tensor([[0.1, 0.2, 0.3]]))
        assert torch.allclose(model_data['current_fill'], torch.tensor([[0.05, 0.15, 0.25]]))

    @pytest.mark.unit
    def test_get_daily_results_with_tour(self, mocker):
        """Test processing daily results when a tour is performed."""
        mock_bins = mocker.MagicMock()
        mock_bins.collect.return_value = (150.0, 2) # 150kg collected, 2 bins
        mock_bins.travel = 0
        
        coordinates = pd.DataFrame({'ID': [0, 101, 102]}) # IDs 0, 1, 2 -> Sim IDs 0, 101, 102
        coordinates.index = [0, 1, 2] # Index
        
        tour = [0, 1, 2, 0] # Bin indices 1 and 2
        cost = 50.0 # km
        day = 3
        new_overflows = 5
        sum_lost = 10.0
        
        bins_out, dlog = get_daily_results(mock_bins, cost, tour, day, new_overflows, sum_lost, coordinates)
        
        assert bins_out.travel == 50.0
        assert dlog['day'] == 3
        assert dlog['overflows'] == 5
        assert dlog['kg_lost'] == 10.0
        assert dlog['kg'] == 150.0
        assert dlog['ncol'] == 2
        assert dlog['km'] == 50.0
        assert dlog['kg/km'] == 3.0 # 150 / 50
        assert dlog['tour'] == [0, 101, 102, 0] # Should map to real IDs

    @pytest.mark.unit
    def test_get_daily_results_no_tour(self, mocker):
        """Test processing daily results when no tour is performed."""
        mock_bins = mocker.MagicMock()
        mock_bins.collect.return_value = (0, 0)
        mock_bins.travel = 100 # Existing travel
        
        coordinates = pd.DataFrame({'ID': [0, 101, 102]})
        
        tour = [0] # No tour
        cost = 0.0 # km
        day = 4
        new_overflows = 8
        sum_lost = 12.0
        
        bins_out, dlog = get_daily_results(mock_bins, cost, tour, day, new_overflows, sum_lost, coordinates)
        
        assert bins_out.travel == 100 # Unchanged
        assert dlog['day'] == 4
        assert dlog['overflows'] == 8
        assert dlog['kg_lost'] == 12.0
        assert dlog['kg'] == 0
        assert dlog['ncol'] == 0
        assert dlog['km'] == 0
        assert dlog['kg/km'] == 0
        assert dlog['tour'] == [0]
    
    @pytest.mark.unit
    def test_stochastic_filling(self, mock_run_day_deps):
        """Test that stochasticFilling is called when bins are stochastic."""
        mock_run_day_deps['bins'].is_stochastic.return_value = True
        
        # Use a policy that runs quickly, e.g., policy_regular
        run_day(
            graph_size=3, pol='policy_regular3_gamma1', bins=mock_run_day_deps['bins'], 
            new_data=mock_run_day_deps['new_data'], coords=mock_run_day_deps['coords'], 
            run_tsp=True, sample_id=0, overflows=0, day=1, 
            model_env=mock_run_day_deps['model_env'], 
            model_ls=mock_run_day_deps['model_ls'], n_vehicles=1, area='test_area', 
            waste_type='test_waste', distpath_tup=mock_run_day_deps['distpath_tup'], 
            current_collection_day=1, cached=None, device='cpu'
        )
        
        mock_run_day_deps['bins'].stochasticFilling.assert_called_once()
        mock_run_day_deps['bins'].loadFilling.assert_not_called()

    @pytest.mark.unit
    def test_load_filling(self, mock_run_day_deps):
        """Test that loadFilling is called when bins are not stochastic."""
        mock_run_day_deps['bins'].is_stochastic.return_value = False
        
        run_day(
            graph_size=3, pol='policy_regular3_gamma1', bins=mock_run_day_deps['bins'], 
            new_data=mock_run_day_deps['new_data'], coords=mock_run_day_deps['coords'], 
            run_tsp=True, sample_id=0, overflows=0, day=5, 
            model_env=mock_run_day_deps['model_env'], 
            model_ls=mock_run_day_deps['model_ls'], n_vehicles=1, area='test_area', 
            waste_type='test_waste', distpath_tup=mock_run_day_deps['distpath_tup'], 
            current_collection_day=1, cached=None, device='cpu'
        )
        
        mock_run_day_deps['bins'].loadFilling.assert_called_once_with(4) # day - 1
        mock_run_day_deps['bins'].stochasticFilling.assert_not_called()

    @pytest.mark.unit
    def test_policy_last_minute_and_path_invalid_cf(self, mock_run_day_deps):
        """Test 'policy_last_minute_and_path' with an invalid cf value."""
        with pytest.raises(ValueError, match='Invalid cf value for policy_last_minute_and_path: 60'):
            run_day(
                graph_size=3, pol='policy_last_minute_and_path60_gamma1', bins=mock_run_day_deps['bins'], 
                new_data=mock_run_day_deps['new_data'], coords=mock_run_day_deps['coords'], 
                run_tsp=True, sample_id=0, overflows=0, day=1, 
                model_env=mock_run_day_deps['model_env'], 
                model_ls=mock_run_day_deps['model_ls'], n_vehicles=1, area='test_area', 
                waste_type='test_waste', distpath_tup=mock_run_day_deps['distpath_tup'], 
                current_collection_day=1, cached=None, device='cpu'
            )

    @pytest.mark.unit
    def test_policy_regular_invalid_lvl(self, mock_run_day_deps):
        """Test 'policy_regular' with an invalid lvl value."""
        with pytest.raises(ValueError, match='Invalid lvl value for policy_regular: 4'):
            run_day(
                graph_size=3, pol='policy_regular4_gamma1', bins=mock_run_day_deps['bins'], 
                new_data=mock_run_day_deps['new_data'], coords=mock_run_day_deps['coords'], 
                run_tsp=True, sample_id=0, overflows=0, day=1, 
                model_env=mock_run_day_deps['model_env'], 
                model_ls=mock_run_day_deps['model_ls'], n_vehicles=1, area='test_area', 
                waste_type='test_waste', distpath_tup=mock_run_day_deps['distpath_tup'], 
                current_collection_day=1, cached=None, device='cpu'
            )

    @pytest.mark.unit
    def test_policy_look_ahead_invalid_config(self, mock_run_day_deps):
        """Test 'policy_look_ahead' with an invalid configuration."""
        with pytest.raises(ValueError, match='Invalid policy_look_ahead configuration'):
            run_day(
                graph_size=3, pol='policy_look_ahead_z_gamma1', bins=mock_run_day_deps['bins'], 
                new_data=mock_run_day_deps['new_data'], coords=mock_run_day_deps['coords'], 
                run_tsp=True, sample_id=0, overflows=0, day=1, 
                model_env=mock_run_day_deps['model_env'], 
                model_ls=mock_run_day_deps['model_ls'], n_vehicles=1, area='test_area', 
                waste_type='test_waste', distpath_tup=mock_run_day_deps['distpath_tup'], 
                current_collection_day=1, cached=None, device='cpu'
            )

    @pytest.mark.unit
    def test_unknown_policy(self, mock_run_day_deps):
        """Test that an unknown policy raises a ValueError."""
        with pytest.raises(ValueError, match='Unknown policy:'):
            run_day(
                graph_size=3, pol='this_policy_does_not_exist_gamma1', bins=mock_run_day_deps['bins'], 
                new_data=mock_run_day_deps['new_data'], coords=mock_run_day_deps['coords'], 
                run_tsp=True, sample_id=0, overflows=0, day=1, 
                model_env=mock_run_day_deps['model_env'], 
                model_ls=mock_run_day_deps['model_ls'], n_vehicles=1, area='test_area', 
                waste_type='test_waste', distpath_tup=mock_run_day_deps['distpath_tup'], 
                current_collection_day=1, cached=None, device='cpu'
            )


class TestSimulation:
    """Tests for the WSmart+ Route simulations."""
    @pytest.mark.unit
    def test_init_single_sim_worker(self, mock_lock_counter):
        """Test that globals are set correctly."""
        mock_lock, mock_counter = mock_lock_counter
        
        # Ensure they are None initially (or don't exist)
        simulation._lock = None
        simulation._counter = None
        
        simulation.init_single_sim_worker(mock_lock, mock_counter)
        
        assert simulation._lock is mock_lock
        assert simulation._counter is mock_counter

    @pytest.mark.unit
    def test_save_matrix_to_excel_exists(self, mocker, tmp_path):
        """Test that to_excel is not called if file already exists."""
        
        # 1. Mock both existence check functions
        mock_exists = mocker.patch('os.path.exists', return_value=True)
        mock_isfile = mocker.patch('os.path.isfile', return_value=True) # Must mock isfile too
        
        # 2. Mock the target function
        mock_to_excel = mocker.patch('pandas.DataFrame.to_excel')
        
        # 3. Call the function
        simulation.save_matrix_to_excel(
            matrix=np.array([[1]]),
            results_dir=str(tmp_path),
            seed=42,
            data_dist='gamma',
            policy='test_pol',
            sample_id=1
        )
        
        # 4. Assert
        mock_to_excel.assert_not_called()
        
        # 5. Verify the correct file path was checked
        expected_path = os.path.join(str(tmp_path), 'fill_history', 'gamma', "enchimentos_seed42_sample1.xlsx")
        mock_exists.assert_any_call(expected_path)
        mock_isfile.assert_called_once_with(expected_path)

    @pytest.mark.unit
    def test_save_matrix_to_excel_new(self, mocker, tmp_path):
        """Test that to_excel is called when file does not exist."""
        mocker.patch('os.path.exists', return_value=False)
        mock_to_excel = mocker.patch('pandas.DataFrame.to_excel')
        
        simulation.save_matrix_to_excel(
            matrix=np.array([[1, 2], [3, 4]]),
            results_dir=str(tmp_path),
            seed=42,
            data_dist='gamma',
            policy='test_pol',
            sample_id=1
        )
        
        mock_to_excel.assert_called_once()
        call_args = mock_to_excel.call_args[0]
        assert call_args[0].endswith(f"test_pol42_sample1.xlsx")

    @pytest.mark.unit
    def test__setup_basedata(self, mocker):
        """Test the base data setup helper."""
        mock_depot_df = pd.DataFrame({'ID': [0]})
        mock_data_df = pd.DataFrame({'ID': [1], 'shape': [(1,1)]}) 
        mock_coords_df = pd.DataFrame({'ID': [1], 'shape': [(1,1)]})
        
        mocker.patch('backend.src.pipeline.simulator.simulation.load_depot', return_value=mock_depot_df)
        mocker.patch('backend.src.pipeline.simulator.simulation.load_simulator_data', 
            return_value=(mock_data_df, mock_coords_df))
        
        data, coords, depot = simulation._setup_basedata(1, 'data_dir', 'area', 'waste')
        
        assert data is mock_data_df
        assert coords is mock_coords_df
        assert depot is mock_depot_df

    @pytest.mark.unit
    def test__setup_dist_path_tup(self, mocker, mock_torch_device):
        """Test the distance/path tuple setup helper."""
        mock_coords = pd.DataFrame({'Lat': [0, 1], 'Lng': [0, 1]})
        mock_dist_matrix = np.array([[0, 10], [10, 0]])
        mock_dist_edges = np.array([[0, 10.0], [10.0, 0]])
        mock_paths = 'mock_shortest_paths'
        mock_adj = np.array([[1, 1], [1, 1]])
        
        mocker.patch('backend.src.pipeline.simulator.simulation.compute_distance_matrix', 
                    return_value=mock_dist_matrix)
        mocker.patch('backend.src.pipeline.simulator.simulation.apply_edges', 
                    return_value=(mock_dist_edges, mock_paths, mock_adj))
        mocker.patch('backend.src.pipeline.simulator.simulation.get_paths_between_states', 
                    return_value='all_paths')
        mock_torch_from_numpy = mocker.patch('torch.from_numpy', 
                                            return_value=torch.tensor(mock_dist_edges))
        
        (dist_tup, adj_matrix) = simulation._setup_dist_path_tup(
            mock_coords, 1, 'hsd', None, None, None, None, mock_torch_device, 50, 'knn'
        )
        
        simulation.compute_distance_matrix.assert_called_once()
        simulation.apply_edges.assert_called_once()
        simulation.get_paths_between_states.assert_called_once_with(2, mock_paths)
        
        assert adj_matrix is mock_adj
        assert dist_tup[0] is mock_dist_edges
        assert dist_tup[1] == 'all_paths'
        assert torch.equal(dist_tup[2], torch.tensor(mock_dist_edges))
        assert np.array_equal(dist_tup[3], (mock_dist_edges * 10).astype('int32'))


    # === Integration Tests for Core Functions ===

    @pytest.mark.integration
    def test_single_simulation_happy_path_am(
        self, wsr_opts, mock_lock_counter, mock_torch_device, mocker
    ):
        """
        Tests the 'happy path' for a 5-day 'am_policy' simulation.
        This test mocks all external dependencies individually.
        """
        # --- 1. Arrange: Set up Options ---
        opts = wsr_opts.copy()
        opts['policies'] = ['am_policy_gamma1']
        opts['days'] = 5

        # Set up global lock/counter
        simulation._lock, simulation._counter = mock_lock_counter

        # --- 2. Arrange: Mock All Dependencies ---

        # Mock data loading to return RAW dataframes
        depot_df = pd.DataFrame({
            'ID': [0], 'Lat': [40.0], 'Lng': [-8.0],
            'Stock': [0], 'Accum_Rate': [0]
        })
        bins_raw_df = pd.DataFrame({'ID': [1,2], 'Lat': [40.1,40.2], 'Lng': [-8.1,-8.2]})
        data_raw_df = pd.DataFrame({'ID': [1,2], 'Stock': [10,20], 'Accum_Rate': [0,0]})
        
        mocker.patch(
            'backend.src.pipeline.simulator.simulation._setup_basedata',
            return_value=(data_raw_df, bins_raw_df, depot_df)
        )

        # Mock the setup_df function (which combines depot + raw data)
        # It will be called twice: 1. for coords, 2. for data
        coords_combined_df = pd.DataFrame({
            'ID': [0, 1, 2], 'Lat': [40.0, 40.1, 40.2], 'Lng': [-8.0, -8.1, -8.2]
        })
        data_combined_df = pd.DataFrame({
            'ID': [0, 1, 2], 'Stock': [0, 10, 20], 'Accum_Rate': [0, 0, 0]
        })
        
        # We mock setup_df where it's *called* (inside processor.py)
        mocker.patch(
            'backend.src.pipeline.simulator.processor.setup_df',
            side_effect=[coords_combined_df, data_combined_df]
        )

        # Mock process_data to be a pass-through
        mocker.patch(
            'backend.src.pipeline.simulator.processor.process_data',
            side_effect=lambda data, bins_coords, depot, indices: (data, bins_coords)
        )

        # Mock network/model setup
        mock_dist_tup = (np.zeros((3,3)), MagicMock(), MagicMock(), np.zeros((3,3)))
        mocker.patch(
            'backend.src.pipeline.simulator.simulation._setup_dist_path_tup',
            return_value=(mock_dist_tup, np.zeros((3,3)))
        )
        mocker.patch(
            'backend.src.utils.setup_utils.setup_model',
            return_value=(MagicMock(), MagicMock()) # Returns (model_env, configs)
        )
        mocker.patch(
            'backend.src.pipeline.simulator.processor.process_model_data',
            return_value=(None, None) # (model_tup)
        )

        # Mock Bins class
        mocker.patch('backend.src.pipeline.simulator.bins.Bins', return_value=MagicMock())

        # Mock checkpointing
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = MagicMock() # Mock the 'hook'
        mocker.patch(
            'backend.src.pipeline.simulator.checkpoints.checkpoint_manager',
            return_value=mock_cm
        )

        # Mock tqdm
        mocker.patch('backend.src.pipeline.simulator.simulation.tqdm', lambda x, **kwargs: x)

        # Mock the core work: run_day
        # This is the key for the assertion
        mock_run_day = mocker.patch(
            'backend.src.pipeline.simulator.day.run_day',
            return_value={'inoverflow': 30} # The expected result
        )

        # --- 3. Act ---
        result = simulation.single_simulation(
            opts, mock_torch_device, indices=None, sample_id=0, pol_id=0,
            model_weights_path=None, n_cores=1
        )

        # --- 4. Assert ---
        assert result['success']
        # The result is a dict, policy name is the key
        # The value is a list of daily results (in this case, the 'inoverflow' value)
        assert result['am_policy_gamma1'] == [30, 30, 30, 30, 30]
        assert mock_run_day.call_count == 5

    @pytest.mark.integration
    def test_single_simulation_resume(
        self, wsr_opts, mock_sim_dependencies, mock_lock_counter, mock_torch_device
    ):
        """Test a simulation that resumes from a checkpoint."""
        opts = wsr_opts
        opts['policies'] = ['policy_gamma1']
        opts['days'] = 10
        opts['resume'] = True

        # Mock a saved state
        mock_saved_state = (
            'mock_data', 'mock_coords', 'mock_dist_tup', 'mock_adj',
            mock_sim_dependencies['bins'], 'mock_model_tup', None, 0, 0, {}, 0
        )
        # Resume from day 5 (so 5 days left to run)
        mock_sim_dependencies['checkpoint'].load_state.return_value = (mock_saved_state, 5) 

        simulation._lock, simulation._counter = mock_lock_counter
        
        result = simulation.single_simulation(
            opts, mock_torch_device, indices=None, sample_id=0, pol_id=0, 
            model_weights_path=None, n_cores=1
        )
        
        # Check setup
        assert mock_sim_dependencies['checkpoint'].load_state.called_once()
        # These should NOT be called on resume
        assert not mock_sim_dependencies['process_data'].called
        assert not mock_sim_dependencies['_setup_dist_path_tup'].called
        
        # Check execution
        # Should run from day 6 to 10 (5 days)
        assert mock_sim_dependencies['run_day'].call_count == 5 
        
        # Check result
        assert result['success']

    @pytest.mark.integration
    def test_single_simulation_checkpoint_error(
        self, wsr_opts, mocker, mock_lock_counter, mock_torch_device
    ):
        """Test that CheckpointError is caught and returned."""
        N = wsr_opts['size'] if 'size' in wsr_opts else 10
        
        # Expected columns: ID, Stock, Accum_Rate (based on loader.py output)
        mock_data = pd.DataFrame({
            'ID': np.arange(1, N + 1),
            'Stock': np.zeros(N),
            'Accum_Rate': np.zeros(N)
        })
        
        mock_coords = pd.DataFrame({
            'ID': np.arange(1, N + 1),
            'Lat': np.zeros(N),
            'Lng': np.zeros(N)
        })
        
        mock_depot = pd.DataFrame({'ID': [0], 'Lat': [0], 'Lng': [0], 'Stock': [0], 'Accum_Rate': [0]})

        mocker.patch(
            'backend.src.pipeline.simulator.simulation._setup_basedata', 
            return_value=(mock_data, mock_coords, mock_depot)
        )

        # The expected return value is ((dist_matrix_edges, paths, dm_tensor, distC), adj_matrix)
        mock_dist_tup = (np.zeros((N+1, N+1)), MagicMock(), MagicMock(), np.zeros((N+1, N+1), dtype='int32'))
        mock_adj_matrix = np.zeros((N+1, N+1))
        mocker.patch(
            'backend.src.pipeline.simulator.simulation._setup_dist_path_tup',
            return_value=(mock_dist_tup, mock_adj_matrix)
        )
        
        # Patch the Bins constructor to return a mock object
        mocker.patch(
            'backend.src.pipeline.simulator.simulation.Bins',
            return_value=MagicMock()
        )

        # Mock the checkpoint manager to raise the error
        error_result = {'success': False, 'error': 'test error'}
        mocker.patch(
            'backend.src.pipeline.simulator.simulation.checkpoint_manager', 
            side_effect=CheckpointError(error_result)
        )
        
        simulation._lock, simulation._counter = mock_lock_counter
        
        result = simulation.single_simulation(
            wsr_opts, mock_torch_device, indices=None, sample_id=0, pol_id=0, 
            model_weights_path=None, n_cores=1
        )
        assert result == error_result

    @pytest.mark.integration
    def test_sequential_simulations_multi_sample(
        self, wsr_opts, mock_sim_dependencies, mock_lock_counter, mock_torch_device
    ):
        """Test sequential simulation with n_samples > 1."""
        opts = wsr_opts
        opts['n_samples'] = 2
        opts['days'] = 5
        opts['policies'] = ['policy_gamma1']
        
        indices_ls = [None, None] # List of indices for 2 samples
        sample_idx_ls = [[0, 1]]  # Policy 0 runs samples 0 and 1
        
        log, log_std, failed = simulation.sequential_simulations(
            opts, mock_torch_device, indices_ls, sample_idx_ls, 
            model_weights_path=None, lock=mock_lock_counter[0]
        )
        
        # Check execution
        # Should run 2 samples * 5 days = 10 times
        assert mock_sim_dependencies['run_day'].call_count == 10
        
        # Check setup (process_data, etc. are called per-sample)
        assert mock_sim_dependencies['process_data'].call_count == 2
        assert mock_sim_dependencies['_setup_dist_path_tup'].call_count == 2
        
        # Check teardown (logging)
        # 2 'full' logs, 2 'daily' logs
        assert mock_sim_dependencies['log_to_json'].call_count == 4 
        
        # Check stats calls
        assert statistics.mean.called
        assert statistics.stdev.called
        
        # Check results
        assert 'policy_gamma1' in log
        assert 'policy_gamma1' in log_std
        assert len(failed) == 0
        assert log['policy_gamma1'][0] == 1.0 # from mock statistics.mean
        assert log_std['policy_gamma1'][0] == 0.1 # from mock statistics.stdev
