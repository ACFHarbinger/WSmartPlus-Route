"""
Fixtures for I/O and data processing unit tests.
"""

import shutil
import tempfile

import pytest


@pytest.fixture
def io_temp_dir():
    """Creates access to a temp dir, cleaning up after."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def sample_dict_data():
    """Sample nested dictionary data."""
    return {"a": {"x": 10}, "b": {"x": 20}}


@pytest.fixture
def sample_list_data():
    """Sample list of dictionaries data."""
    return [{"entry": {"x": 10}}, {"entry": {"x": 20}}]
