
import pytest
from typing import Any, Dict, Iterator, Optional
from logic.src.interfaces.traversable import ITraversable

def test_dict_is_traversable():
    """Verify that a standard dictionary matches the ITraversable protocol."""
    d = {"key": "value"}
    assert isinstance(d, ITraversable)

def test_custom_class_is_traversable():
    """Verify that a custom class implementing required methods matches."""
    class MyTraversable:
        def __getitem__(self, key: Any) -> Any:
            return "value"
        def __contains__(self, key: Any) -> bool:
            return True
        def __iter__(self) -> Iterator[Any]:
            return iter(["key"])
        def __len__(self) -> int:
            return 1
        def keys(self) -> Any:
            return ["key"]
        def items(self) -> Any:
            return [("key", "value")]
        def values(self) -> Any:
            return ["value"]
        def get(self, key: Any, default: Any = None) -> Any:
            return "value"

    obj = MyTraversable()
    assert isinstance(obj, ITraversable)

def test_string_is_not_traversable():
    """Verify that a string (which has __getitem__ and __contains__) is NOT traversable."""
    # ITraversable requires keys(), items(), values(), and get()
    assert not isinstance("string", ITraversable)

def test_list_is_not_traversable():
    """Verify that a list is NOT traversable."""
    assert not isinstance([1, 2, 3], ITraversable)

def test_dataclass_mock_is_traversable():
    """Verify that a class with the same structure as our config dataclasses matches."""
    class MockConfig:
        def __getitem__(self, key: Any) -> Any: return None
        def __contains__(self, key: Any) -> bool: return False
        def __iter__(self) -> Iterator[Any]: return iter([])
        def __len__(self) -> int: return 0
        def keys(self) -> Any: return []
        def items(self) -> Any: return []
        def values(self) -> Any: return []
        def get(self, key: Any, default: Any = None) -> Any: return default

    assert isinstance(MockConfig(), ITraversable)

try:
    from omegaconf import DictConfig
    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False

@pytest.mark.skipif(not HAS_OMEGACONF, reason="OmegaConf not installed")
def test_dictconfig_is_traversable():
    """Verify that OmegaConf DictConfig matches."""
    cfg = DictConfig({"a": 1})
    assert isinstance(cfg, ITraversable)
