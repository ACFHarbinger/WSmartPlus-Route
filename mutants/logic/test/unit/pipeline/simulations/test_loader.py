from unittest.mock import mock_open, patch

import logic.src.constants as udef
import pandas as pd
import pytest
from logic.src.pipeline.simulations.repository import (
    FileSystemRepository,
    _repository,
    load_area_and_waste_type_params,
)


class TestFileSystemRepository:
    @pytest.fixture
    def repo(self, tmp_path):
        return FileSystemRepository(str(tmp_path))

    def test_wrappers(self):
        # Test that wrappers call the imported function
        with patch("logic.src.utils.data.data_utils.load_area_and_waste_type_params") as mock_method:
            load_area_and_waste_type_params("A", "B")
            mock_method.assert_called_with("A", "B")
