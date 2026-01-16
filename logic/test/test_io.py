import json
import os
import zipfile

import pandas as pd

from logic.src.utils.io.preview import (
    preview_changes,
    preview_file_changes,
    preview_file_statistics,
)
from logic.src.utils.io.processing import (
    find_single_input_values,
    find_two_input_values,
    process_dict_of_dicts,
    process_file,
    process_list_of_dicts,
)
from logic.src.utils.io.splitting import chunk_zip_content, reassemble_files, split_file


class TestIOSplitting:
    def test_split_and_reassemble_csv(self, io_temp_dir):
        data_dir = os.path.join(io_temp_dir, "data")
        os.makedirs(data_dir)

        # Create a dummy CSV file
        df = pd.DataFrame({"col1": range(100), "col2": range(100)})
        file_path = os.path.join(data_dir, "test.csv")
        df.to_csv(file_path, index=False)

        # Split
        # Force small chunks by mocking memory usage logic or just utilizing the size parameter
        # Since logic uses memory_usage, we need to pick a small max_part_size
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

        # force split with small max_part_size
        files = chunk_zip_content(zip_path, max_part_size=50, data_dir=data_out)

        assert len(files) > 1
        assert any("inside_part" in f for f in files)

        # Reassemble from the extracted dir
        reassemble_files(data_out)
        assert os.path.exists(os.path.join(data_out, "inside.csv"))


class TestIOProcessing:
    def test_process_dict_of_dicts(self, sample_dict_data):
        # scale logic: val * update_val + val
        def func(old, update):
            return old * 2 + update

        # With update_val
        process_dict_of_dicts(sample_dict_data, output_key="x", process_func=func, update_val=5)
        # 10 * 2 + 5 = 25
        assert sample_dict_data["a"]["x"] == 25

        # With list
        data2 = {"a": {"x": [10, 20]}}
        process_dict_of_dicts(data2, output_key="x", process_func=func, update_val=5)
        assert data2["a"]["x"] == [25, 45]

    def test_process_list_of_dicts(self, sample_list_data):
        def func(old, update):
            return old + update

        process_list_of_dicts(sample_list_data, output_key="x", process_func=func, update_val=1.0)
        # 10 + 1 = 11
        assert sample_list_data[0]["entry"]["x"] == 11.0

    def test_find_input_values(self):
        d = {"k": 5}
        val = find_single_input_values(d, output_key="k")
        assert val[0][1] == 5

        d2 = {"k1": 5, "k2": 10}
        val = find_two_input_values(d2, input_key1="k1", input_key2="k2")
        assert val[0][1] == 5
        assert val[0][2] == 10


class TestIOPreview:
    def test_preview_changes(self, io_temp_dir):
        json_path = os.path.join(io_temp_dir, "log_1.json")
        with open(json_path, "w") as f:
            json.dump({"entry": {"km": 100}}, f)

        def stats(x, keys):
            return x

        # Should print something, returning None is fine
        preview_changes(io_temp_dir, output_key="km", update_val=0.5, process_func=stats)

    def test_preview_file_changes(self, io_temp_dir):
        json_path = os.path.join(io_temp_dir, "log_1.json")
        if not os.path.exists(json_path):
            with open(json_path, "w") as f:
                json.dump({"entry": {"km": 100}}, f)

        def stats(x, keys):
            return x

        preview_file_changes(json_path, output_key="km", update_val=0.5, process_func=stats)

    def test_preview_statistics(self, io_temp_dir):
        json_path = os.path.join(io_temp_dir, "log_1.json")
        if not os.path.exists(json_path):
            with open(json_path, "w") as f:
                json.dump({"entry": {"km": 100}}, f)

        # Create a stats function
        def stats(x):
            return sum(x)

        preview_file_statistics(json_path, output_key="km", process_func=stats)


class TestIOIntegration:
    def test_process_file(self, io_temp_dir):
        json_path = os.path.join(io_temp_dir, "test.json")
        with open(json_path, "w") as f:
            # Nested structure
            json.dump({"entry": {"val": 10}}, f)

        def func(old, val):
            return old + val

        process_file(json_path, output_key="val", update_val=1.0, process_func=func)
        with open(json_path) as f:
            d = json.load(f)
        # 10 + 1 = 11
        assert d["entry"]["val"] == 11.0
