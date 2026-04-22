"""
Utility to aggregate simulation results JSONs into a single Excel log.

Attributes:
    _DIST_PATTERN: Regex pattern to extract distribution information from policy names.
    _DISPLAY_METRICS: List of metrics to display in the summary.
    SELECTED_DIRS: Whitelist of directories to include in the summary.

Example:
    >>> from logic.src.utils.output.excel_summary import discover_and_aggregate
    >>> df = discover_and_aggregate()
    >>> print(df.head())
"""

import json
import os
import re
from typing import Any, Dict, List, Tuple

import pandas as pd

from logic.src.constants import ROOT_DIR

# Constants
_DIST_PATTERN = re.compile(r"_(emp|gamma\d*|uniform)$", re.IGNORECASE)
_DISPLAY_METRICS = [
    "profit",
    "cost",
    "kg",
    "km",
    "kg/km",
    "overflows",
    "ncol",
    "kg_lost",
    "days",
    "time",
]

# Whitelist of directories to include in the summary.
# If empty, all directories under assets/output/ are included.
# Example: ["31_days/riomaior_104", "30_days/riomaior_104"]
SELECTED_DIRS: List[str] = [
    "31_days/riomaior_104",
    "30_days/riomaior_104",
]


def _parse_policy_name(raw_name: str) -> Tuple[str, str]:
    """
    Split a raw policy key into (base_name, distribution).

    Args:
        raw_name:  Raw policy key.

    Returns:
        Tuple[str, str]: A tuple containing the base name and distribution.
    """
    match = _DIST_PATTERN.search(raw_name)
    if match:
        dist = match.group(1).lower()
        base = raw_name[: match.start()].rstrip("_")
        return base, dist
    return raw_name, "unknown"


def _load_json(path: str) -> Any:
    """
    Load and parse a JSON file.

    Args:
        path:  Path to the JSON file.

    Returns:
        Any: The parsed JSON data.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def discover_and_aggregate() -> pd.DataFrame:
    """
    Find all log directories and aggregate results into a DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the aggregated simulation results.
    """
    output_root = os.path.join(ROOT_DIR, "assets", "output")
    all_rows: List[Dict[str, Any]] = []

    if not os.path.isdir(output_root):
        print(f"Directory not found: {output_root}")
        return pd.DataFrame()

    for root, _, files in os.walk(output_root):
        rel_path = os.path.relpath(root, output_root)

        # Skip if not in whitelist (if whitelist is provided)
        if SELECTED_DIRS and not any(rel_path.startswith(d) for d in SELECTED_DIRS):
            continue

        mean_file = next((f for f in files if "log_mean" in f.lower() and f.endswith(".json")), None)
        if not mean_file:
            continue

        std_file = next((f for f in files if "log_std" in f.lower() and f.endswith(".json")), None)

        rel_path = os.path.relpath(root, output_root)
        mean_path = os.path.join(root, mean_file)
        std_path = os.path.join(root, std_file) if std_file else None

        mean_data = _load_json(mean_path)
        std_data = _load_json(std_path) if std_path else None

        if not isinstance(mean_data, dict):
            continue

        for policy_key, metrics in mean_data.items():
            if not isinstance(metrics, dict):
                continue

            base_name, dist = _parse_policy_name(policy_key)
            row: Dict[str, Any] = {
                "SourceDir": rel_path,
                "Policy": base_name,
                "Distribution": dist,
                "Policy_Key": policy_key,
            }

            for m, v in metrics.items():
                row[f"{m}_mean"] = v
                if std_data and policy_key in std_data:
                    row[f"{m}_std"] = std_data[policy_key].get(m, 0.0)
                else:
                    row[f"{m}_std"] = 0.0

            all_rows.append(row)

    return pd.DataFrame(all_rows)


def main() -> None:
    """Main execution entry point."""
    print("Collecting simulation results...")
    df = discover_and_aggregate()

    if df.empty:
        print("No simulation data found.")
        return

    output_path = os.path.join(ROOT_DIR, "assets", "output", "simulation_summary.xlsx")
    print(f"Found {len(df)} policy entries. Exporting to Excel...")

    # Sort for readability
    cols_priority = ["SourceDir", "Distribution", "Policy"]
    df = df.sort_values(cols_priority)

    # Save to Excel
    try:
        df.to_excel(output_path, index=False)
        print(f"SUCCESS: Summary saved to {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to save Excel file: {e}")


if __name__ == "__main__":
    main()
