"""Tests for config_loader.py."""

import pytest
from logic.src.utils.configs.config_loader import load_config


def test_load_yaml_config(tmp_path):
    """Verify YAML configuration loading."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("a: 1\nb: [2, 3]")

    config = load_config(str(yaml_path))
    assert config["a"] == 1
    assert config["b"] == [2, 3]


def test_load_xml_config(tmp_path):
    """Verify XML configuration loading."""
    xml_path = tmp_path / "config.xml"
    xml_path.write_text("<config><item>1</item><list>2</list><list>3</list></config>")

    config = load_config(str(xml_path))
    assert config["item"] == 1
    assert config["list"] == [2, 3]


def test_load_config_not_found():
    """Verify error handling for missing config file."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_file.yaml")


def test_load_config_invalid_extension(tmp_path):
    """Verify error handling for unsupported file extensions."""
    txt_path = tmp_path / "config.txt"
    txt_path.write_text("some text")
    with pytest.raises(ValueError, match="Unsupported config file extension"):
        load_config(str(txt_path))
