"""Unit tests for data utility functions (data_utils.py)."""

import pytest
import tempfile
import os
import pickle
from unittest.mock import MagicMock, patch
import torch
from tensordict import TensorDict
from logic.src.utils.data.data_utils import (
    check_extension,
    save_dataset,
    load_dataset,
    save_td_dataset,
    load_td_dataset,
    collate_fn,
    generate_waste_prize,
    load_area_and_waste_type_params,
)


class TestCheckExtension:
    def test_adds_extension_if_missing(self):
        result = check_extension("file", ".pkl")
        assert result == "file.pkl"

    def test_keeps_extension_if_present(self):
        result = check_extension("file.pkl", ".pkl")
        assert result == "file.pkl"

    def test_different_extension(self):
        result = check_extension("file", ".td")
        assert result == "file.td"


class TestSaveLoadDataset:
    def test_save_and_load_roundtrip(self, tmp_path):
        data = {"key": [1, 2, 3], "tensor": torch.randn(5, 5)}
        filepath = str(tmp_path / "test.pkl")

        save_dataset(data, filepath)
        loaded = load_dataset(filepath)

        assert loaded["key"] == [1, 2, 3]
        assert torch.allclose(loaded["tensor"], data["tensor"])

    def test_save_creates_directory(self, tmp_path):
        data = [1, 2, 3]
        filepath = str(tmp_path / "subdir" / "test.pkl")

        save_dataset(data, filepath)
        assert os.path.exists(filepath)


class TestTensorDictDataset:
    def test_save_and_load_td(self, tmp_path):
        td = TensorDict({"a": torch.randn(3, 4)}, batch_size=[3])
        filepath = str(tmp_path / "td_test")

        save_td_dataset(td, filepath)
        loaded = load_td_dataset(filepath)

        assert loaded.batch_size == td.batch_size
        assert torch.allclose(loaded["a"], td["a"])


class TestCollateFn:
    def test_filters_none_values(self):
        batch = [{"a": 1}, None, {"a": 2}]
        result = collate_fn(batch)
        assert len(result) == 2
        assert result == [{"a": 1}, {"a": 2}]

    def test_empty_batch(self):
        result = collate_fn([None, None])
        assert result == []


class TestGenerateWastePrize:
    def test_empty_distribution(self):
        result = generate_waste_prize(10, "empty", (MagicMock(), MagicMock()))
        assert result.shape == (1, 10)
        assert torch.all(result == 0)

    def test_const_distribution(self):
        result = generate_waste_prize(10, "const", (MagicMock(), MagicMock()))
        assert result.shape == (1, 10)
        assert torch.all(result == 1)

    def test_uniform_distribution(self):
        result = generate_waste_prize(10, "unif", (MagicMock(), MagicMock()), dataset_size=5)
        assert result.shape == (5, 10)
        assert torch.all((result >= 0) & (result <= 1))

    def test_gamma_distribution(self):
        result = generate_waste_prize(10, "gamma1", (MagicMock(), MagicMock()))
        assert result.shape == (1, 10)


class TestLoadAreaAndWasteTypeParams:
    def test_mixrmbac_glass(self):
        capacity, maxfill, dumping, binvol, ndays = load_area_and_waste_type_params("mixrmbac", "glass")
        assert capacity > 0
        assert maxfill > 0
        assert binvol > 0

    def test_mixrmbac_paper(self):
        capacity, maxfill, dumping, binvol, ndays = load_area_and_waste_type_params("mixrmbac", "paper")
        assert capacity > 0

    def test_invalid_area(self):
        with pytest.raises(AssertionError):
            load_area_and_waste_type_params("invalid_area", "glass")

    def test_invalid_waste(self):
        with pytest.raises(AssertionError):
            load_area_and_waste_type_params("mixrmbac", "invalid_waste")
