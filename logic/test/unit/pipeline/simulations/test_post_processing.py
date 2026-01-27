"""Tests for Route Post-Processing sub-system."""

import numpy as np
import pytest
from logic.src.policies.post_processing import (
    PostProcessorFactory,
    FastTSPPostProcessor,
    ClassicalLocalSearchPostProcessor,
    RandomLocalSearchPostProcessor
)

@pytest.fixture
def sample_route_data():
    """Sample data for post-processing tests."""
    dist_matrix = np.array([
        [0, 10, 20, 10],
        [10, 0, 10, 20],
        [20, 10, 0, 10],
        [10, 20, 10, 0],
    ], dtype=np.int32)
    tour = [0, 2, 1, 3, 0] # Inefficient tour
    return dist_matrix, tour

class TestPostProcessorFactory:
    def test_factory_get_fast_tsp(self):
        proc = PostProcessorFactory.create("fast_tsp")
        assert isinstance(proc, FastTSPPostProcessor)

    def test_factory_get_classical_ls(self):
        proc = PostProcessorFactory.create("2opt")
        assert isinstance(proc, ClassicalLocalSearchPostProcessor)

    def test_factory_invalid(self):
        with pytest.raises(ValueError):
            PostProcessorFactory.create("invalid_processor")

class TestFastTSPPostProcessor:
    def test_fast_tsp_refinement(self, sample_route_data):
        dist_matrix, tour = sample_route_data
        processor = FastTSPPostProcessor()
        refined_tour = processor.process(tour, distance_matrix=dist_matrix)
        assert len(refined_tour) == len(tour)
        assert refined_tour[0] == 0
        assert refined_tour[-1] == 0

class TestClassicalLocalSearchPostProcessor:
    def test_2opt_refinement(self, sample_route_data):
        dist_matrix, tour = sample_route_data
        processor = ClassicalLocalSearchPostProcessor(operator_name="2opt")
        refined_tour = processor.process(tour, distance_matrix=dist_matrix)
        assert len(refined_tour) == len(tour)
        assert refined_tour[0] == 0
        assert refined_tour[-1] == 0

class TestPathPostProcessor:
    def test_path_refinement_fills_gap(self):
        # Scenario: Tour 0 -> 1 -> 3 -> 0
        # Path 1->3 goes through 2: [1, 2, 3]
        # Bin 2 is not in tour. fits capacity.

        processor = PostProcessorFactory.create("path")

        tour = [0, 1, 3, 0]

        # Mock paths: paths[1][3] = [1, 2, 3]
        # We need a structure that supports paths[u][v]
        # Minimal mock using dict/list
        paths = [[[] for _ in range(4)] for _ in range(4)]
        paths[1][3] = [1, 2, 3]

        # Mock fill levels: bin 1=10, bin 2=10, bin 3=10
        # Indices in fill array are bin_id-1.
        # bin 1 -> idx 0, etc.
        current_fill = np.array([10.0, 10.0, 10.0, 10.0]) # 4 bins

        refined = processor.process(
            tour,
            paths_between_states=paths,
            total_fill=current_fill,
            vehicle_capacity=100.0,
            bins=None
        )

        # Expect 2 to be inserted between 1 and 3
        expected = [0, 1, 2, 3, 0]
        assert refined == expected

    def test_path_refinement_skip_if_full(self):
        processor = PostProcessorFactory.create("path")
        tour = [0, 1, 3, 0]
        paths = [[[] for _ in range(4)] for _ in range(4)]
        paths[1][3] = [1, 2, 3]
        current_fill = np.array([50.0, 50.0, 50.0, 50.0])

        # Current load: 1(50) + 3(50) = 100. Capacity = 100.
        # Node 2(50) + 100 = 150 > 100. Should skip.

        refined = processor.process(
            tour,
            paths_between_states=paths,
            total_fill=current_fill,
            vehicle_capacity=100.0
        )

        assert refined == tour

    def test_path_refinement_skip_existing(self):
        # If 2 is already in tour, don't double add
        processor = PostProcessorFactory.create("path")
        tour = [0, 1, 2, 3, 0]
        paths = [[[] for _ in range(4)] for _ in range(4)]
        paths[1][3] = [1, 2, 3]
        current_fill = np.array([10.0, 10.0, 10.0, 10.0])

        refined = processor.process(
            tour,
            paths_between_states=paths,
            total_fill=current_fill,
            vehicle_capacity=100.0
        )

        assert refined == tour
