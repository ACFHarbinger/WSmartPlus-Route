"""
Configuration utilities module.

Attributes:
    load_yaml_config: Load a YAML configuration file.
    load_xml_config: Load an XML configuration file.
    load_config: Load a configuration file.
    to_bash_value: Convert a Python value to a Bash-friendly string.
    load_yaml_env: Load a YAML configuration file and recursively merge its defaults.
    deep_merge: Deeply merge source dictionary into target dictionary.

Example:
    load_yaml_config("path/to/config.yaml")
    load_xml_config("path/to/config.xml")
    load_config("path/to/config.yaml")
    to_bash_value("value")
    load_yaml_env("path/to/config.yaml")
    deep_merge(target, source)
"""

from .config_loader import load_yaml_config, load_xml_config, load_config
from .yaml_to_env import to_bash_value, load_yaml_env, deep_merge

__all__ = [
    "load_yaml_config",
    "load_xml_config",
    "load_config",
    "to_bash_value",
    "load_yaml_env",
    "deep_merge",
]
