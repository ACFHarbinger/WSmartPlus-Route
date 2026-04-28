"""
Search space definitions for simulation policies.

This module contains the default hyperparameter search spaces for all supported
simulation policies, including ALNS, HGS, ACO, and SISR. Each policy's search
space is defined as a dictionary with parameter names as keys and their HPO
specifications as values.

Attributes:
    POLICY_SEARCH_SPACES: Dictionary mapping policy names to their search spaces.

Functions:
    get_search_space: Retrieves the search space for a specific policy.

Example:
    >>> from logic.src.policies.helpers.hpo import get_search_space
    >>> alns_space = get_search_space("alns")
    >>> print(alns_space)
    {
        'max_iterations': {'type': 'int', 'low': 500, 'high': 10000, 'step': 500},
        'start_temp': {'type': 'float', 'low': 1.0, 'high': 500.0, 'log': True},
        ...
    }
"""

import dataclasses
import importlib
import inspect
import json
import os
import pkgutil
from typing import Any, Dict

# Path to the directory containing JSON mandatory node selection strategies parameter search spaces definitions
FILTERS_DIR = os.path.join(os.path.dirname(__file__), "filters")

# Path to the directory containing JSON route improvers parameter search spaces definitions
INTERCEPTORS_DIR = os.path.join(os.path.dirname(__file__), "interceptors")

# Path to the directory containing JSON route constructors parameter search spaces definitions
JOBS_DIR = os.path.join(os.path.dirname(__file__), "jobs")

# Path to the directory containing JSON move acceptance criteria parameter search spaces definitions
RULES_DIR = os.path.join(os.path.dirname(__file__), "rules")


def load_all_search_spaces() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load all search space definitions from the jobs/ directory.

    Returns:
        Dict[str, Dict[str, Dict[str, Any]]]: Mapping of policy names to search spaces.
    """
    spaces: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not os.path.exists(JOBS_DIR):
        return spaces

    for filename in os.listdir(JOBS_DIR):
        if filename.endswith(".json"):
            policy_name = filename[:-5].lower()
            try:
                with open(os.path.join(JOBS_DIR, filename), "r") as f:
                    spaces[policy_name] = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                # In a production environment, we might want to log this error
                print(f"Error loading route constructor parameters search space from {filename}: {e}")

    return spaces


# Default search spaces loaded from external JSON files
POLICY_SEARCH_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = load_all_search_spaces()


def get_search_space(policy_name: str) -> Dict[str, Dict[str, Any]]:
    """Get the search space for a specific policy.

    Args:
        policy_name (str): Name of the policy (e.g., 'alns', 'hgs').

    Returns:
        Dict[str, Dict[str, Any]]: The search space specification dictionary for
            the requested policy, or an empty dictionary if not found.
    """
    return POLICY_SEARCH_SPACES.get(policy_name.lower(), {})


def _extract_params_from_config(config_class: Any) -> Dict[str, Any]:
    """Extract parameters and their basic types from a dataclass config.

    Args:
        config_class (Any): The dataclass configuration class to inspect.

    Returns:
        Dict[str, Any]: A dictionary mapping field names to their HPO hints (type, default).
    """
    params = {}
    for field in dataclasses.fields(config_class):
        # Basic type detection for HPO hints
        field_type = str(field.type).lower()
        if "bool" in field_type:
            t = "categorical"
            choices = [True, False]
        elif "int" in field_type:
            t = "int"
            choices = None
        elif "float" in field_type:
            t = "float"
            choices = None
        elif "str" in field_type:
            t = "categorical"
            choices = [field.default] if field.default is not dataclasses.MISSING else []
        else:
            t = "object"
            choices = None

        param_spec: Dict[str, Any] = {"type": t}
        if choices is not None:
            param_spec["choices"] = choices

        # Add metadata about default value if JSON serializable
        if field.default is not dataclasses.MISSING:
            try:
                json.dumps(field.default)
                param_spec["default"] = field.default
            except (TypeError, OverflowError):
                param_spec["default"] = str(field.default)
        elif field.default_factory is not dataclasses.MISSING:
            param_spec["default"] = "factory"

        params[field.name] = param_spec
    return params


def generate_policy_filters() -> None:
    """Read all policy configs and create JSON filters with all parameters in the filters directory.

    This function scans the logic.src.configs.policies package, identifies dataclass
    configurations, extracts their fields, and saves them as JSON templates in the
    directory specified by JOBS_DIR.
    """
    from logic.src.configs import policies as policies_pkg

    os.makedirs(JOBS_DIR, exist_ok=True)
    package_path = policies_pkg.__path__

    for _, module_name, is_pkg in pkgutil.walk_packages(package_path):
        if is_pkg or module_name in ("abc", "__init__"):
            continue

        full_module_name = f"logic.src.configs.policies.{module_name}"
        try:
            module = importlib.import_module(full_module_name)
        except Exception as e:
            print(f"Error importing {full_module_name}: {e}")
            continue

        # Find the main Config class in the module
        config_class = None
        for name, obj in inspect.getmembers(module):
            if (
                name.endswith("Config")
                and dataclasses.is_dataclass(obj)
                and getattr(obj, "__module__", "") == full_module_name
            ):
                config_class = obj
                break

        if config_class:
            # Handle nested policies and remove 'config' suffixes
            base_name = module_name.split(".")[-1].lower().replace("config", "").replace("postconfig", "")
            params = _extract_params_from_config(config_class)

            output_path = os.path.join(JOBS_DIR, f"{base_name}.json")
            with open(output_path, "w") as f:
                json.dump(params, f, indent=4)
            print(f"Generated route constructor parameter search space: {output_path}")


def generate_route_improvement_interceptors() -> None:
    """Read all route improvement configs and create JSON interceptors in the interceptors directory.

    This function scans logic.src.configs.policies.other.route_improvement, identifies
    dataclass configurations, extracts their fields, and saves them as JSON templates
    in the directory specified by INTERCEPTORS_DIR.
    """
    from logic.src.configs.policies.other import route_improvement

    os.makedirs(INTERCEPTORS_DIR, exist_ok=True)

    # Find all dataclasses in the module
    for name, obj in inspect.getmembers(route_improvement):
        if (
            (name.endswith("Config") or name.endswith("PostConfig"))
            and dataclasses.is_dataclass(obj)
            and getattr(obj, "__module__", "") == route_improvement.__name__
        ):
            # Strip suffixes to get a clean name (e.g., 'FastTSPPostConfig' -> 'fasttsp')
            base_name = name.lower().replace("postconfig", "").replace("config", "")
            params = _extract_params_from_config(obj)

            output_path = os.path.join(INTERCEPTORS_DIR, f"{base_name}.json")
            with open(output_path, "w") as f:
                json.dump(params, f, indent=4)
            print(f"Generated route improver parameter search space: {output_path}")


def generate_mandatory_selection_jobs() -> None:
    """Read all mandatory selection configs and create JSON jobs in the jobs directory.

    This function scans logic.src.configs.policies.other.mandatory_selection, identifies
    dataclass configurations, extracts their fields, and saves them as JSON templates
    in the directory specified by FILTERS_DIR.
    """
    from logic.src.configs.policies.other import mandatory_selection

    os.makedirs(FILTERS_DIR, exist_ok=True)

    # Find all dataclasses in the module
    for name, obj in inspect.getmembers(mandatory_selection):
        if (
            name.endswith("Config")
            and dataclasses.is_dataclass(obj)
            and getattr(obj, "__module__", "") == mandatory_selection.__name__
        ):
            # Strip suffixes to get a clean name (e.g., 'LastMinuteSelectionConfig' -> 'lastminute')
            base_name = name.lower().replace("selectionconfig", "").replace("postconfig", "").replace("config", "")
            params = _extract_params_from_config(obj)

            output_path = os.path.join(FILTERS_DIR, f"{base_name}.json")
            with open(output_path, "w") as f:
                json.dump(params, f, indent=4)
            print(f"Generated mandatory node selection search space: {output_path}")


def generate_acceptance_criteria_rules() -> None:
    """Read all acceptance criteria configs and create JSON rules in the rules directory.

    This function scans logic.src.configs.policies.other.acceptance_criteria, identifies
    dataclass configurations, extracts their fields, and saves them as JSON templates
    in the directory specified by RULES_DIR.
    """
    from logic.src.configs.policies.other import acceptance_criteria

    os.makedirs(RULES_DIR, exist_ok=True)

    # Find all dataclasses in the module
    for name, obj in inspect.getmembers(acceptance_criteria):
        if (
            name.endswith("Config")
            and dataclasses.is_dataclass(obj)
            and getattr(obj, "__module__", "") == acceptance_criteria.__name__
        ):
            # Strip suffixes to get a clean name (e.g., 'BoltzmannAcceptanceConfig' -> 'boltzmann')
            base_name = (
                name.lower()
                .replace("acceptanceconfig", "")
                .replace("selectionconfig", "")
                .replace("postconfig", "")
                .replace("config", "")
            )
            params = _extract_params_from_config(obj)

            output_path = os.path.join(RULES_DIR, f"{base_name}.json")
            with open(output_path, "w") as f:
                json.dump(params, f, indent=4)
            print(f"Generated rule filter: {output_path}")


if __name__ == "__main__":
    generate_policy_filters()
    generate_route_improvement_interceptors()
    generate_mandatory_selection_jobs()
    generate_acceptance_criteria_rules()
