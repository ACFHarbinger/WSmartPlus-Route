"""
Comprehensive test suite for I/O utilities.

This module consolidates tests for:
- File operations (io/files.py)
- Locking utilities (io/locking.py)
- Data processing (io/processing.py)
- File splitting and reassembly (io/splitting.py)
- Preview utilities (io/preview.py)
"""

import json
import os
import tempfile
import zipfile

import pandas as pd

from logic.src.utils.io import files, locking, processing
from logic.src.utils.io.preview import (
    preview_changes,
    preview_file_changes,
    preview_file_statistics,
)
from logic.src.utils.io.splitting import chunk_zip_content, reassemble_files, split_file

# ============================================================================
# File Operations Tests
# ============================================================================


class TestFiles:
    """Tests for file operation utilities."""

    def test_read_json(self):
        """Test JSON file reading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = os.path.join(tmp_dir, "test.json")
            data = {"key": "value"}
            with open(json_path, "w") as f:
                json.dump(data, f)

            assert files.read_json(json_path) == data

    def test_zip_extract_directory(self):
        """Test directory zipping and extraction."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a directory to zip
            input_dir = os.path.join(tmp_dir, "input")
            os.makedirs(input_dir)
            with open(os.path.join(input_dir, "file1.txt"), "w") as f:
                f.write("hello")

            zip_path = os.path.join(tmp_dir, "test.zip")
            files.zip_directory(input_dir, zip_path)
            assert os.path.exists(zip_path)

            # Extract
            output_dir = os.path.join(tmp_dir, "output")
            files.extract_zip(zip_path, output_dir)
            assert os.path.exists(os.path.join(output_dir, "file1.txt"))
            with open(os.path.join(output_dir, "file1.txt"), "r") as f:
                assert f.read() == "hello"

    def test_compose_dirpath(self):
        """Test directory path composition decorator."""

        @files.compose_dirpath
        def my_func(path, extra=None):
            return path, extra

        home = "/root"
        days = 10
        nbins = 50
        area = "city"

        # Single nbins
        path, extra = my_func(home, days, nbins, "data", area, extra="yes")
        expected_path = os.path.join(home, "assets", "data", "10_days", "city_50")
        assert path == expected_path
        assert extra == "yes"

        # Multiple nbins
        nbins_list = [50, 100]
        paths, extra = my_func(home, days, nbins_list, "data", area, extra="no")
        assert isinstance(paths, list)
        assert len(paths) == 2
        assert paths[0] == os.path.join(home, "assets", "data", "10_days", "city_50")
        assert paths[1] == os.path.join(home, "assets", "data", "10_days", "city_100")
        assert extra == "no"


# ============================================================================
# Locking Utilities Tests
# ============================================================================


class TestLocking:
    """Tests for I/O locking utilities."""

    def test_read_output(self):
        """Test reading and transposing policy output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = os.path.join(tmp_dir, "test.json")
            data = {"policy1": [10, 20, 30], "policy2": [5, 15, 25], "ignored": [0, 0, 0]}
            with open(json_path, "w") as f:
                json.dump(data, f)

            policies = ["policy1", "policy2"]
            res = locking.read_output(json_path, policies)
            assert res == [[10, 5], [20, 15], [30, 25]]


# ============================================================================
# Data Processing Tests
# ============================================================================


class TestProcessing:
    """Tests for data processing utilities."""

    def test_process_dict_of_dicts(self):
        """Test processing nested dictionaries."""
        data = {"node1": {"km": 10, "other": 5}, "node2": {"km": [1, 2, 3], "other": 0}}
        modified = processing.process_dict_of_dicts(data, "km", lambda x, _: x * 2)
        assert modified is True
        assert data["node1"]["km"] == 20
        assert data["node2"]["km"] == [2, 4, 6]

    def test_process_list_of_dicts(self):
        """Test processing list of dictionaries."""
        data = [{"node1": {"km": 10}}, {"node2": {"km": 20}}]
        modified = processing.process_list_of_dicts(data, "km", lambda x, u: x + u, update_val=5)
        assert modified is True
        assert data[0]["node1"]["km"] == 15
        assert data[1]["node2"]["km"] == 25

    def test_process_dict_two_inputs(self):
        """Test processing with two input keys."""
        data = {"node1": {"km": 10, "miles": 6.2}, "node2": {"km": 20, "miles": 12.4}}
        modified = processing.process_dict_two_inputs(data, "km", "miles", "sum", lambda v1, v2: v1 + v2)
        assert modified is True
        assert abs(data["node1"]["sum"] - 16.2) < 1e-6

    def test_find_single_input_values(self):
        """Test finding values by key in nested structure."""
        data = {"a": {"km": 10}, "b": [{"km": 20}, {"km": 30}]}
        res = processing.find_single_input_values(data, output_key="km")
        assert len(res) == 3
        values = [v for p, v in res]
        assert 10 in values
        assert 20 in values
        assert 30 in values

    def test_process_file(self):
        """Test file processing with modification."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "test.json")
            data = {"a": {"km": 10}}
            with open(file_path, "w") as f:
                json.dump(data, f)

            success = processing.process_file(file_path, "km", lambda x, _: x + 1)
            assert success is True

            with open(file_path, "r") as f:
                new_data = json.load(f)
            assert new_data["a"]["km"] == 11


# ============================================================================
# File Splitting Tests
# ============================================================================


class TestSplitting:
    """Tests for file splitting and reassembly."""

    def test_split_and_reassemble_csv(self, io_temp_dir):
        """Test CSV file splitting and reassembly."""
        data_dir = os.path.join(io_temp_dir, "data")
        os.makedirs(data_dir)

        # Create a dummy CSV file
        df = pd.DataFrame({"col1": range(100), "col2": range(100)})
        file_path = os.path.join(data_dir, "test.csv")
        df.to_csv(file_path, index=False)

        # Split
        chunks = split_file(file_path, max_part_size=100, output_dir=data_dir)
        assert len(chunks) > 1, "File should have been split into multiple chunks"

        # Cleanup original file to verify reassembly
        os.remove(file_path)

        # Reassemble
        reassembled = reassemble_files(data_dir)
        assert "test.csv" in reassembled
        assert os.path.exists(file_path)

        # Verify content
        df_new = pd.read_csv(file_path)
        pd.testing.assert_frame_equal(df, df_new)

    def test_chunk_zip_content(self, io_temp_dir):
        """Test chunking zip file contents."""
        data_dir = os.path.join(io_temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # Create dummy file inside zip
        csv_path = os.path.join(data_dir, "inside.csv")
        df = pd.DataFrame({"x": range(100)})
        df.to_csv(csv_path, index=False)

        zip_path = os.path.join(io_temp_dir, "test.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(csv_path, arcname="inside.csv")

        os.remove(csv_path)

        # Chunk zip content
        data_out = os.path.join(io_temp_dir, "extracted")
        os.makedirs(data_out)

        files_list = chunk_zip_content(zip_path, max_part_size=50, data_dir=data_out)
        assert len(files_list) > 1
        assert any("inside_part" in f for f in files_list)

        # Reassemble from the extracted dir
        reassemble_files(data_out)
        assert os.path.exists(os.path.join(data_out, "inside.csv"))


# ============================================================================
# Preview Utilities Tests
# ============================================================================


class TestPreview:
    """Tests for preview utilities."""

    def test_preview_changes(self, io_temp_dir):
        """Test previewing changes to directory."""
        json_path = os.path.join(io_temp_dir, "log_1.json")
        with open(json_path, "w") as f:
            json.dump({"entry": {"km": 100}}, f)

        def stats(x, keys):
            return x

        preview_changes(io_temp_dir, output_key="km", update_val=0.5, process_func=stats)

    def test_preview_file_changes(self, io_temp_dir):
        """Test previewing changes to single file."""
        json_path = os.path.join(io_temp_dir, "log_1.json")
        if not os.path.exists(json_path):
            with open(json_path, "w") as f:
                json.dump({"entry": {"km": 100}}, f)

        def stats(x, keys):
            return x

        preview_file_changes(json_path, output_key="km", update_val=0.5, process_func=stats)

    def test_preview_statistics(self, io_temp_dir):
        """Test previewing file statistics."""
        json_path = os.path.join(io_temp_dir, "log_1.json")
        if not os.path.exists(json_path):
            with open(json_path, "w") as f:
                json.dump({"entry": {"km": 100}}, f)

        def stats(x):
            return sum(x)

        preview_file_statistics(json_path, output_key="km", process_func=stats)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIOIntegration:
    """Integration tests for I/O utilities."""

    def test_process_file_integration(self, io_temp_dir):
        """Test end-to-end file processing."""
        json_path = os.path.join(io_temp_dir, "test.json")
        with open(json_path, "w") as f:
            json.dump({"entry": {"val": 10}}, f)

        def func(old, val):
            return old + val

        processing.process_file(json_path, output_key="val", update_val=1.0, process_func=func)

        with open(json_path) as f:
            d = json.load(f)
        assert d["entry"]["val"] == 11.0
