"""
Fixtures for Policy Unit Tests.
"""


import numpy as np
import pytest


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
    distancesC = np.array(
        [
            [0, 10, 10, 15, 20, 15],  # Depot 0
            [10, 0, 5, 10, 15, 10],  # Bin 1
            [10, 5, 0, 10, 15, 10],  # Bin 2
            [15, 10, 10, 0, 5, 5],  # Bin 3
            [20, 15, 15, 5, 0, 5],  # Bin 4
            [15, 10, 10, 5, 5, 0],  # Bin 5
        ],
        dtype=np.int32,
    )

    # Paths for policy_last_minute_and_path (Nodes 0-5)
    paths_between_states = [
        [[]],  # 0
        [[1, 0], [], [1, 2], [1, 5, 3], [1, 5, 4], [1, 5]],  # 1
        [[2, 0], [2, 1], [], [2, 1, 5, 3], [2, 1, 5, 4], [2, 1, 5]],  # 2
        [[3, 5, 0], [3, 5, 1], [3, 5, 1, 2], [], [3, 4], [3, 5]],  # 3
        [[4, 5, 0], [4, 5, 1], [4, 5, 1, 2], [4, 3], [], [4, 5]],  # 4
        [[5, 0], [5, 1], [5, 1, 2], [5, 3], [5, 4], []],  # 5
    ]

    # 2. Mock dependent functions
    mock_load_params = mocker.patch(
        "logic.src.pipeline.simulations.loader.load_area_and_waste_type_params",
        return_value=(4000, 0.16, 21.0, 1.0, 2.5),  # Q, R, B, C, V
    )
    mocker.patch("logic.src.policies.regular.load_area_and_waste_type_params", mock_load_params)
    mocker.patch(
        "logic.src.policies.last_minute.load_area_and_waste_type_params",
        mock_load_params,
    )

    # Mock TSP solver (used by last_minute and regular)
    mock_find_route = mocker.patch(
        "logic.src.policies.single_vehicle.find_route",
        return_value=[0, 1, 3, 0],  # Default mock tour
    )
    mocker.patch("logic.src.policies.regular.find_route", mock_find_route)
    mocker.patch("logic.src.policies.last_minute.find_route", mock_find_route)

    # Mock multi-tour splitter
    mock_get_multi_tour = mocker.patch(
        "logic.src.policies.single_vehicle.get_multi_tour",
        side_effect=lambda tour, *args: tour,  # Pass-through
    )
    mocker.patch("logic.src.policies.regular.get_multi_tour", mock_get_multi_tour)
    mocker.patch("logic.src.policies.last_minute.get_multi_tour", mock_get_multi_tour)

    return {
        "n_bins": 5,
        "bins_waste": bins_waste,
        "distancesC": distancesC,
        "paths_between_states": paths_between_states,
        "mocks": {
            "load_params": mock_load_params,
            "find_route": mock_find_route,
            "get_multi_tour": mock_get_multi_tour,
        },
    }


@pytest.fixture
def mock_policy_common_data():
    """Provides common data structures (distances, waste, paths) for policy unit tests."""
    # 5 bins + 1 depot (node 0)
    distancesC = np.array(
        [
            [0, 10, 10, 15, 20, 15],  # Depot 0
            [10, 0, 5, 10, 15, 10],  # Bin 1 (idx 1)
            [10, 5, 0, 10, 15, 10],  # Bin 2 (idx 2)
            [15, 10, 10, 0, 5, 5],  # Bin 3 (idx 3)
            [20, 15, 15, 5, 0, 5],  # Bin 4 (idx 4)
            [15, 10, 10, 5, 5, 0],  # Bin 5 (idx 5)
        ],
        dtype=np.int32,
    )

    # Fill levels for bins 1-5 (indices 0-4)
    bins_waste = np.array([10.0, 95.0, 30.0, 85.0, 50.0])

    # Mock paths for 'last_minute_and_path' testing (full 6x6 node structure)
    paths_between_states = [
        [[]] * 6,
        [[]] * 6,
        [[2, 0], [2, 1], [2], [2, 1, 5, 3], [2, 1, 5, 4], [2, 1, 5]],  # Example paths
        [[]] * 6,
        [[]] * 6,
        [[]] * 6,
    ]

    return {
        "n_bins": 5,
        "bins_waste": bins_waste,
        "distancesC": distancesC,
        "distance_matrix": distancesC.astype(float),
        "paths_between_states": paths_between_states,
    }


@pytest.fixture
def mock_policy_dependencies(mocker):
    """Mocks common policy dependencies (loader, solver) for unit tests."""
    # Mock TSP solver (used by last_minute)
    mocker.patch(
        "logic.src.policies.single_vehicle.find_route",
        return_value=[0, 1, 3, 0],  # Default mock tour for 2 bins
    )
    # Mock multi-tour splitter
    mocker.patch(
        "logic.src.policies.single_vehicle.get_multi_tour",
        side_effect=lambda tour, *args: tour,  # Pass-through
    )
    # Mock distance matrix used by single_vehicle helpers
    mocker.patch("logic.src.policies.single_vehicle.get_route_cost", return_value=50.0)


@pytest.fixture
def mock_lookahead_aux(mocker):
    """Mocks the internal look_ahead_aux dependencies."""

    # Mocks must be autospec=True to be bound correctly to the look_ahead module import
    mocker.patch(
        "logic.src.policies.look_ahead.should_bin_be_collected",
        autospec=True,
        side_effect=lambda fill, rate: fill + rate >= 100,
    )
    mocker.patch(
        "logic.src.policies.look_ahead.update_fill_levels_after_first_collection",
        autospec=True,
        # Returns fill levels after collected bins are reset
        return_value=np.array([0.0, 10.0, 30.0, 40.0, 50.0]),
    )
    mocker.patch(
        "logic.src.policies.look_ahead.get_next_collection_day",
        autospec=True,
        return_value=5,  # Mocked next overflow day
    )
    mocker.patch(
        "logic.src.policies.look_ahead.add_bins_to_collect",
        autospec=True,
        return_value=[0, 1, 2],  # Mocked final bin list
    )


@pytest.fixture
def mock_vrpp_inputs(mock_policy_common_data):
    """Provides data structures needed for VRPP policies."""
    data = mock_policy_common_data

    # Mock predicted values (media + param * std)
    media = np.full(data["n_bins"], 10.0)
    std = np.full(data["n_bins"], 1.0)

    return {
        "bins": data["bins_waste"],  # [10.0, 95.0, 30.0, 85.0, 50.0]
        "distances": data["distance_matrix"].tolist(),  # Use float matrix
        "distance_matrix": data["distance_matrix"],
        "media": media,
        "std": std,
        "must_go_bins": [1, 3],  # Bin indices (0-indexed)
        "binsids": list(range(data["n_bins"])),  # [0, 1, 2, 3, 4]
    }


@pytest.fixture
def mock_optimizer_data(mock_policy_common_data):
    """Provides data structures needed for VRPP policies."""
    data = mock_policy_common_data

    # Mock predicted values (media + param * std)
    media = np.full(data["n_bins"], 10.0)
    std = np.full(data["n_bins"], 1.0)

    return {
        "bins": data["bins_waste"],  # [10.0, 95.0, 30.0, 85.0, 50.0]
        "distances": data["distance_matrix"].tolist(),  # Use float matrix
        "media": media,
        "std": std,
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
        [40, 30, 20, 10, 0],
    ]

    demands = {1: 10, 2: 10, 3: 10, 4: 10}
    capacity = 100
    R = 1.0
    C = 1.0

    global_must_go = {2, 4}  # Must go bins
    local_to_global = {0: 1, 1: 2, 2: 3, 3: 4}  # Linear mapping

    vrpp_tour_global = [2, 4, 1, 3]  # Some initial tour

    return (
        dist_matrix,
        demands,
        capacity,
        R,
        C,
        global_must_go,
        local_to_global,
        vrpp_tour_global,
    )
