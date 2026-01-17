"""
Comprehensive test suite for utility modules.

This module consolidates tests for:
- Graph utilities (graph_utils.py)
- General utility functions (functions.py)
- Cryptography utilities (crypto_utils.py)
- Logging utilities (log_utils.py)
- Neural network utilities (boolmask.py, beam_search.py)
- Debug utilities (debug_utils.py)
- Monkey patches (monkey_patch.py)
- I/O locking (io/locking.py)
- Configuration loading (config_loader.py)
"""

import json
import os
import pickle
import tempfile
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from cryptography.fernet import Fernet

import logic.src.utils.monkey_patch as monkey_patch
from logic.src.utils import beam_search, boolmask, log_utils
from logic.src.utils import crypto_utils as cu
from logic.src.utils.config_loader import load_config
from logic.src.utils.debug_utils import watch
from logic.src.utils.functions import (
    compute_in_batches,
    do_batch_rep,
    get_inner_model,
    get_path_until_string,
    load_args,
    load_problem,
    move_to,
    parse_softmax_temperature,
)

# Import modules to test
from logic.src.utils.graph_utils import (
    adj_to_idx,
    find_longest_path,
    generate_adj_matrix,
    get_adj_knn,
    get_edge_idx_dist,
    idx_to_adj,
    sort_by_pairs,
    tour_to_adj,
)
from logic.src.utils.io.locking import read_output

# ============================================================================
# Graph Utilities Tests
# ============================================================================


class TestGraphUtils:
    """Tests for graph utility functions."""

    def test_generate_adj_matrix(self):
        """Test adjacency matrix generation."""
        # Test basic generation
        adj = generate_adj_matrix(size=5, num_edges=5, add_depot=False)
        assert adj.shape == (5, 5)

        # Test float ratio
        adj_ratio = generate_adj_matrix(size=5, num_edges=0.5, add_depot=False)
        assert adj_ratio.shape == (5, 5)

        # Test undirected
        adj_u = generate_adj_matrix(size=5, num_edges=3, undirected=True, add_depot=False)
        assert np.all(adj_u == adj_u.T)

    def test_get_edge_idx_dist(self):
        """Test edge index extraction from distance matrix."""
        dist_matrix = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
        edge_idx = get_edge_idx_dist(dist_matrix, num_edges=2, add_depot=False, undirected=True)
        assert edge_idx.shape[1] == 2
        # (0,1) and (0,2) are closest
        assert (edge_idx == np.array([[0, 0], [1, 2]])).all() or (edge_idx == np.array([[0, 0], [2, 1]])).all()

    def test_sort_by_pairs(self):
        """Test sorting edge indices by pairs."""
        edge_idx = torch.tensor([[1, 0, 2], [2, 1, 0]])
        sorted_idx = sort_by_pairs(3, edge_idx)
        assert sorted_idx[0, 0] == 0 and sorted_idx[1, 0] == 1
        assert sorted_idx[0, 1] == 1 and sorted_idx[1, 1] == 2
        assert sorted_idx[0, 2] == 2 and sorted_idx[1, 2] == 0

    def test_get_adj_knn(self):
        """Test k-nearest neighbor adjacency matrix."""
        dist_mat = np.array([[0, 1, 2], [1, 0, 0.5], [2, 0.5, 0]])
        adj = get_adj_knn(dist_mat, k_neighbors=1, add_depot=False, negative=False)
        assert adj[1, 2] == 1
        assert adj[1, 0] == 0

    def test_roundtrip_adj_idx(self):
        """Test adjacency matrix to index conversion and back."""
        adj = np.array([[0, 1], [1, 0]])
        idx = adj_to_idx(adj, negative=False)
        adj_back = idx_to_adj(idx, negative=False)
        assert np.array_equal(adj, adj_back)

    def test_tour_to_adj(self):
        """Test tour to adjacency matrix conversion."""
        tour = [0, 2, 1]
        adj = tour_to_adj(tour)
        assert adj[0, 2] == 1 and adj[2, 0] == 1
        assert adj[2, 1] == 1 and adj[1, 2] == 1
        assert adj[1, 0] == 1 and adj[0, 1] == 1

    def test_find_longest_path(self):
        """Test longest path finding in DAG."""
        dist = torch.full((3, 3), float("-inf"))
        dist[0, 1] = 5
        dist[1, 2] = 10
        dist[0, 2] = 7
        length, path = find_longest_path(dist, start_vertex=0)
        assert length == 15
        assert path == [0, 1, 2]


# ============================================================================
# General Functions Tests
# ============================================================================


class TestFunctions:
    """Tests for general utility functions."""

    def test_get_inner_model(self):
        """Test extracting inner model from DataParallel wrapper."""
        model = nn.Linear(10, 2)
        assert get_inner_model(model) == model

        wrapper = nn.DataParallel(model)
        assert get_inner_model(wrapper) == model

    def test_load_problem(self):
        """Test problem class loading."""
        from logic.src.problems.vrpp.problem_vrpp import VRPP

        assert load_problem("vrpp") == VRPP

        with pytest.raises(AssertionError):
            load_problem("invalid_problem_name_123")

    def test_move_to(self):
        """Test moving tensors/dicts to device."""
        device = torch.device("cpu")
        t = torch.randn(5)
        moved = move_to(t, device)
        assert moved.device.type == "cpu"

        d = {"a": torch.randn(2), "b": [torch.randn(3)]}
        moved_d = move_to(d, device)
        assert isinstance(moved_d, dict)
        assert moved_d["a"].device.type == "cpu"
        assert moved_d["b"][0].device.type == "cpu"

        assert move_to(None, device) is None
        assert move_to([1, 2], device) == [1, 2]

    def test_load_args(self, tmp_path):
        """Test loading arguments from JSON file."""
        arg_file = tmp_path / "args.json"
        data = {"problem": "vrpp_dist", "other": 1}
        arg_file.write_text(json.dumps(data))
        args = load_args(str(arg_file))
        assert args["problem"] == "vrpp"
        assert args["data_distribution"] == "dist"
        assert args["aggregation_graph"] == "avg"

    def test_parse_softmax_temperature(self):
        """Test softmax temperature parsing."""
        assert parse_softmax_temperature(2.0) == 2.0
        assert parse_softmax_temperature("1.5") == 1.5

    def test_get_path_until_string(self):
        """Test path extraction until substring."""
        path = "/home/user/project/logic/src/file.py"
        assert get_path_until_string(path, "logic") == "/home/user/project/logic"
        assert get_path_until_string(path, "nonexistent") is None

    def test_do_batch_rep(self):
        """Test batch replication."""
        t = torch.randn(2, 3)
        rep = do_batch_rep(t, 2)
        assert rep.shape == (4, 3)

        d = {"x": t}
        rep_d = do_batch_rep(d, 2)
        assert rep_d["x"].shape == (4, 3)

    def test_compute_in_batches(self):
        """Test batched computation."""

        def f(x):
            return x * 2

        x = torch.arange(10)
        res = compute_in_batches(f, 3, x)
        assert torch.equal(res, x * 2)

        def f_multi(x, y):
            return x + y, x - y

        y = torch.arange(10)
        res_plus, res_minus = compute_in_batches(f_multi, 4, x, y)
        assert torch.equal(res_plus, x + y)
        assert torch.equal(res_minus, x - y)


# ============================================================================
# Cryptography Tests
# ============================================================================


class TestCryptoUtils:
    """Tests for cryptography utilities."""

    def test_encode_data(self):
        """Test data encoding to bytes."""
        assert isinstance(cu.encode_data("test"), bytes)
        assert isinstance(cu.encode_data(123), bytes)
        assert isinstance(cu.encode_data(12.34), bytes)
        assert isinstance(cu.encode_data([1, 2]), bytes)

    def test_encrypt_decrypt_file_data(self, tmp_path):
        """Test file encryption and decryption."""
        key = Fernet.generate_key()
        data = "Secret Data"

        encrypted = cu.encrypt_file_data(key, data)
        assert encrypted != data.encode()

        decrypted = cu.decrypt_file_data(key, encrypted)
        assert decrypted == data

        # Test file I/O
        in_file = tmp_path / "input.txt"
        in_file.write_text(data)
        out_file = tmp_path / "output.enc"

        cu.encrypt_file_data(key, str(in_file), str(out_file))
        assert out_file.exists()

        dec_file = tmp_path / "decrypted.txt"
        cu.decrypt_file_data(key, str(out_file), str(dec_file))
        assert dec_file.read_text() == data

    def test_encrypt_decrypt_directory(self, tmp_path):
        """Test directory encryption and decryption."""
        key = Fernet.generate_key()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file1.txt").write_text("content1")
        (input_dir / "subdir").mkdir()
        (input_dir / "subdir" / "file2.txt").write_text("content2")

        output_dir = tmp_path / "encrypted"
        cu.encrypt_directory(key, str(input_dir), str(output_dir))

        assert (output_dir / "file1.txt.enc").exists()
        assert (output_dir / "subdir" / "file2.txt.enc").exists()

        decrypted_dir = tmp_path / "decrypted"
        cu.decrypt_directory(key, str(output_dir), str(decrypted_dir))

        assert (decrypted_dir / "file1.txt").read_text() == "content1"
        assert (decrypted_dir / "subdir" / "file2.txt").read_text() == "content2"


# ============================================================================
# Logging Utilities Tests
# ============================================================================


class TestLogUtils:
    """Tests for logging utilities."""

    def test_get_loss_stats(self):
        """Test loss statistics calculation."""
        epoch_loss = {
            "loss1": [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])],
            "loss2": [torch.tensor([10.0])],
        }
        stats = log_utils.get_loss_stats(epoch_loss)
        assert len(stats) == 8
        assert stats[0] == 2.0
        assert stats[4] == 10.0

    def test_convert_numpy(self):
        """Test numpy to Python type conversion."""
        assert log_utils._convert_numpy(np.array([1, 2])) == [1, 2]
        assert log_utils._convert_numpy(np.float32(1.0)) == 1.0
        assert log_utils._convert_numpy(np.int64(5)) == 5
        assert log_utils._convert_numpy({"a": np.float32(1.0)}) == {"a": 1.0}

    def test_sort_log(self):
        """Test log sorting."""
        log = {"policies": ["p1", "p2"], "p2": [{"id": 2}], "p1": [{"id": 1}]}
        sorted_log = log_utils._sort_log(log)
        keys = list(sorted_log.keys())
        assert "policies" in keys
        assert "p1" in keys
        assert "p2" in keys

    def test_log_to_json2(self):
        """Test thread-safe JSON logging."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = os.path.join(tmp_dir, "test.json")
            keys = ["metric1"]
            dit = {"policy1": {"val": 100}}

            result = log_utils.log_to_json2(json_path, keys, dit, sort_log=False)

            assert os.path.exists(json_path)
            assert "policy1" in result

    def test_log_to_pickle(self):
        """Test pickle logging."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pickle_path = os.path.join(tmp_dir, "test.pkl")
            log_data = {"a": 1, "b": 2}
            log_utils.log_to_pickle(pickle_path, log_data)
            assert os.path.exists(pickle_path)

            with open(pickle_path, "rb") as f:
                loaded = pickle.load(f)
            assert loaded == log_data


# ============================================================================
# Neural Utilities Tests
# ============================================================================


class TestBoolMask:
    """Tests for boolean mask utilities."""

    def test_pad_mask(self):
        """Test mask padding."""
        mask = torch.ones(7, dtype=torch.uint8)
        padded, n_bytes = boolmask._pad_mask(mask)
        assert padded.size(-1) == 8
        assert n_bytes == 1

        mask2 = torch.ones(8, dtype=torch.uint8)
        padded2, n_bytes2 = boolmask._pad_mask(mask2)
        assert padded2.size(-1) == 8
        assert n_bytes2 == 1

    def test_bool2byte2long2bool(self):
        """Test mask conversion roundtrip."""
        mask = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype=torch.uint8)

        byte_mask = boolmask._mask_bool2byte(mask)
        long_mask = boolmask._mask_byte2long(byte_mask)
        recovered_bool = boolmask.mask_long2bool(long_mask, n=10)

        assert torch.equal(mask.bool(), recovered_bool)

    def test_mask_long_scatter(self):
        """Test mask scatter operation."""
        mask = torch.zeros((2, 3, 1), dtype=torch.int64)
        values = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)

        updated = boolmask.mask_long_scatter(mask, values)
        assert updated[0, 0, 0].item() == (1 << 1)
        assert updated[1, 2, 0].item() == (1 << 6)


class TestBeamSearch:
    """Tests for beam search utilities."""

    @pytest.fixture
    def mock_state(self):
        """Create mock state for beam search testing."""

        def create_mock_state(ids, finished=False):
            state = MagicMock()
            state.ids = ids
            state.get_mask.return_value = torch.zeros((len(ids), 1, 3), dtype=torch.uint8)
            state.all_finished.return_value = finished
            state.get_final_cost.return_value = torch.zeros((len(ids), 1))
            state.construct_solutions.return_value = [[0, 1, 0]] * len(ids)

            def get_item(key):
                try:
                    new_ids = state.ids[key]
                except Exception:
                    new_ids = state.ids
                return create_mock_state(new_ids, finished=finished)

            state.__getitem__.side_effect = get_item

            def update_fn(action):
                return create_mock_state(state.ids, finished=True)

            state.update.side_effect = update_fn
            state.to.return_value = state
            return state

        return create_mock_state(torch.arange(2))

    def test_beam_search_basic(self, mock_state):
        """Test basic beam search functionality."""

        def expanded_proposals(beam):
            return torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0.9, 0.8])

        res = beam_search.beam_search(mock_state, beam_size=1, propose_expansions=expanded_proposals)
        score, solutions, cost, ids, batch_size = res

        assert batch_size == 2
        assert len(solutions) == 2
        assert isinstance(solutions, list)

    def test_segment_topk_idx(self):
        """Test segment-wise top-k selection."""
        x = torch.tensor([10.0, 5.0, 20.0, 1.0, 30.0], dtype=torch.float)
        ids = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)

        idx = beam_search.segment_topk_idx(x, 1, ids)
        assert set(idx.tolist()) == {0, 4}

    def test_backtrack(self):
        """Test backtracking through beam search history."""
        parents = [torch.tensor([0, 1]), torch.tensor([0, 1])]
        actions = [torch.tensor([10, 20]), torch.tensor([100, 200])]

        res = beam_search.backtrack(parents, actions)
        expected = torch.tensor([[10, 100], [20, 200]])
        assert torch.equal(res, expected)

    def test_cached_lookup(self):
        """Test cached lookup functionality."""
        data = torch.randn(10, 5)
        lookup = beam_search.CachedLookup(data)

        idx = torch.tensor([0, 1, 2])
        res1 = lookup[idx]
        assert res1 is lookup.current

        res2 = lookup[idx]
        assert res2 is res1  # cached

        idx2 = torch.tensor([3, 4])
        res3 = lookup[idx2]
        assert res3 is not res1
        assert res3 is lookup.current


# ============================================================================
# Debug & System Utilities Tests
# ============================================================================


class TestDebugUtils:
    """Tests for debug utilities."""

    def test_watch_simple_variable(self):
        """Test variable watching functionality."""
        val = 1
        with patch("sys.settrace") as mock_trace:
            watch("val")
            assert val == 1
            assert mock_trace.called


class TestMonkeyPatch:
    """Tests for monkey patch utilities."""

    def test_load_state_dict_cast(self):
        """Test patched load_state_dict."""
        optimizer = MagicMock()
        optimizer.__setstate__ = MagicMock()
        optimizer.param_groups = [{"params": [MagicMock()]}]
        optimizer.param_groups[0]["params"][0].device = "cpu"
        optimizer.param_groups[0]["params"][0].data = torch.tensor([1.0])

        state_dict = {"param_groups": [{"params": [0]}], "state": {0: torch.tensor([2.0], device="cpu")}}

        monkey_patch.load_state_dict(optimizer, state_dict)
        optimizer.__setstate__.assert_called()


class TestLocking:
    """Tests for I/O locking utilities."""

    def test_read_output_with_lock(self):
        """Test reading output with lock."""
        lock = MagicMock()
        mock_data = '{"policy1": [1, 2], "policy2": [3, 4]}'

        with patch("builtins.open", mock_open(read_data=mock_data)):
            with patch("json.load", return_value=json.loads(mock_data)):
                result = read_output("dummy.json", ["policy1"], lock)

        assert lock.acquire.called
        assert lock.release.called
        assert result == [[1], [2]]

    def test_read_output_no_lock(self):
        """Test reading output without lock."""
        mock_data = '{"policy1": [1, 2]}'
        with patch("builtins.open", mock_open(read_data=mock_data)):
            with patch("json.load", return_value=json.loads(mock_data)):
                result = read_output("dummy.json", ["policy1"], None)
        assert result == [[1], [2]]


class TestConfigLoader:
    """Tests for configuration loading."""

    def test_load_yaml(self, tmp_path):
        """Test YAML config loading."""
        f = tmp_path / "test.yaml"
        f.write_text("key: value\nlist: [1, 2]")

        cfg = load_config(str(f))
        assert cfg["key"] == "value"
        assert cfg["list"] == [1, 2]

    def test_load_xml(self, tmp_path):
        """Test XML config loading."""
        f = tmp_path / "test.xml"
        xml_content = """<config>
    <key>value</key>
    <nested>
        <item>1</item>
    </nested>
</config>"""
        f.write_text(xml_content)

        cfg = load_config(str(f))
        assert cfg["key"] == "value"
        assert cfg["nested"]["item"] == 1

    def test_load_config_not_found(self):
        """Test config file not found error."""
        with pytest.raises(FileNotFoundError):
            load_config("non_existent.yml")

    def test_load_config_value_error(self, tmp_path):
        """Test unsupported config format error."""
        f = tmp_path / "test.txt"
        f.touch()
        with pytest.raises(ValueError, match="Unsupported config"):
            load_config(str(f))

    def test_load_yaml_error(self, tmp_path):
        """Test YAML parsing error."""
        f = tmp_path / "bad.yaml"
        f.write_text("key: [unclosed list")
        with pytest.raises(ValueError):
            load_config(str(f))
