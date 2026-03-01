from unittest.mock import mock_open, patch

import logic.src.constants as udef
import pandas as pd
import pytest
from logic.src.pipeline.simulations.repository import (
    FileSystemRepository,
    load_area_and_waste_type_params,
)


class TestFileSystemRepository:
    @pytest.fixture
    def repo(self, tmp_path):
        return FileSystemRepository(str(tmp_path))

    @pytest.fixture(autouse=True)
    def setup_repo(self, repo):
        from logic.src.pipeline.simulations.repository import set_repository
        set_repository(repo)

    def test_wrappers(self):
        # Test that wrappers call the imported function
        with patch("logic.test.unit.pipeline.simulations.test_loader.load_area_and_waste_type_params") as mock_method:
            load_area_and_waste_type_params("riomaior", "glass")
            mock_method.assert_called_with("riomaior", "glass")
