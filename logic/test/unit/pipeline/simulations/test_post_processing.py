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
