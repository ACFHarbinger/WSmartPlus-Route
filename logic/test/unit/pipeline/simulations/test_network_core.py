import pytest
import numpy as np
from logic.src.utils.graph.network_utils import apply_edges, get_paths_between_states

class TestNetwork:
    """Class for network and distance matrix tests."""

    @pytest.mark.unit
    def test_apply_edges_knn(self):
        """Test applying KNN edge thresholds to a distance matrix."""
        # 3x3 matrix (1 depot + 2 bins)
        dm = np.array([
            [0, 10, 20],
            [10, 0, 30],
            [20, 30, 0]
        ], dtype=float)

        # apply_edges sparsifies the bins (indices 1:)
        # We need to provide valid params for knn
        new_dm, paths, adj = apply_edges(dm, edge_thresh=1, edge_method="knn")

        assert isinstance(new_dm, np.ndarray)
        assert new_dm.shape == (3, 3)
        if paths is not None:
            assert isinstance(paths, dict)

    @pytest.mark.unit
    def test_get_paths_between_states(self):
        """Test constructing nested path lists."""
        shortest_paths = {(0, 1): [0, 1], (1, 0): [1, 0]}
        paths = get_paths_between_states(n_bins=2, shortest_paths=shortest_paths)

        assert len(paths) == 2
        assert len(paths[0]) == 2
        assert paths[0][1] == [0, 1]
        assert paths[1][0] == [1, 0]
        assert paths[0][0] == [0, 0] # Self-path
