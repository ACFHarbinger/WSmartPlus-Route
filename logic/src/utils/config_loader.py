"""
Configuration loading utilities.

This module provides functions for:
- Loading YAML configuration files.
- Loading XML configuration files and converting them to dictionaries.
"""

import os
import yaml
import xml.etree.ElementTree as ET


def load_yaml_config(config_path: str) -> dict:
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML file.

    Returns:
        dict: The configuration dictionary.

    Raises:
        ValueError: If parsing fails.
    """
    with open(config_path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")


def load_xml_config(config_path: str) -> dict:
    """
    Loads an XML configuration file and converts it to a dictionary.

    Args:
        config_path (str): Path to the XML file.

    Returns:
        dict: The configuration dictionary.

    Raises:
        ValueError: If parsing fails.
    """
    try:
        tree = ET.parse(config_path)
        root = tree.getroot()

        def xml_to_dict(element):
            """
            Helper to convert XML element to dictionary.

            Args:
                element: The XML element.

            Returns:
                dict or list or str/int/float: The parsed data.
            """
            # Helper to convert XML element to dictionary
            if len(element) == 0:
                try:
                    return int(element.text)
                except (ValueError, TypeError):
                    try:
                        return float(element.text)
                    except (ValueError, TypeError):
                        return element.text

            result = {}
            for child in element:
                child_data = xml_to_dict(child)
                if child.tag in result:
                    if isinstance(result[child.tag], list):
                        result[child.tag].append(child_data)
                    else:
                        result[child.tag] = [result[child.tag], child_data]
                else:
                    result[child.tag] = child_data
            return result

        data = xml_to_dict(root)
        if root.tag == "config" and isinstance(data, dict):
            return data
        return {root.tag: data}

    except ET.ParseError as e:
        raise ValueError(f"Error parsing XML file: {e}")


def load_config(config_path: str) -> dict:
    """
    Loads a YAML or XML configuration file based on extension.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: The configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported or parsing fails.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    _, ext = os.path.splitext(config_path)

    config = {}
    if ext.lower() == ".yaml" or ext.lower() == ".yml":
        config = load_yaml_config(config_path)
        if config is None:
            config = {}
        return config

    elif ext.lower() == ".xml":
        config = load_xml_config(config_path)
        if config is None:
            config = {}
        return config

    else:
        raise ValueError(f"Unsupported config file extension: {ext}")
