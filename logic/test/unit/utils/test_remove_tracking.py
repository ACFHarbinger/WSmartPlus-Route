"""Unit tests for remove_tracking.py functions."""

from pathlib import Path

import pytest
from logic.package.remove_tracking import (
    append_to_class_body,
    fix_empty_try_blocks,
    get_project_root,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]




@pytest.mark.unit
@pytest.mark.fast
def test_get_project_root():
    root = get_project_root()
    assert isinstance(root, Path)
    assert (root / "pyproject.toml").exists() or (root / "logic").exists()


@pytest.mark.unit
@pytest.mark.fast
def test_fix_empty_try_blocks():
    content = (
        "try:\n"
        "    # import logic.src.tracking\n"
        "except ImportError:\n"
        "    pass\n"
    )
    res = fix_empty_try_blocks(content)
    assert "pass" in res
    assert "try:" in res


@pytest.mark.unit
@pytest.mark.fast
def test_append_to_class_body():
    code = (
        "class MyClass:\n"
        "    def existing_method(self):\n"
        "        pass\n"
    )
    method_lines = "def _viz_record(self, **kwargs):\n    pass"
    new_code = append_to_class_body(code, "MyClass", method_lines)
    assert "_viz_record" in new_code
    assert "class MyClass:" in new_code
