"""
Log parsing and data management for simulation results.
"""

import json
from typing import Any, Dict, List, Optional, Set, Tuple


class SimulationDataManager:
    """
    Manages the accumulation and organization of simulation log data.
    """

    def __init__(self, policy_names: List[str]):
        self.policy_names = policy_names
        self.accumulated_data: Dict[str, Dict[str, List[Any]]] = {}
        self.policy_samples: Dict[str, Set[str]] = {}
        self.metrics: Set[str] = set()
        self.day_data: Dict[str, Dict[int, Any]] = {}  # key -> day -> data
        self.history_buffer = ""

    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single log line and return the record if valid.
        """
        line = line.strip()
        if not line:
            return None

        # Check for our structural markers (e.g., specific JSON start)
        if not (line.startswith("{") and line.endswith("}")):
            return None

        try:
            record = json.loads(line)
            return record
        except json.JSONDecodeError:
            return None

    def process_record(self, record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Process a log record and update internal data structures.

        Returns:
            Tuple: (target_key, policy_name, sample_id) if data was updated.
        """
        policy = record.get("policy")
        sample = record.get("sample", "0")
        day = record.get("day", 0)

        if not policy:
            return None, None, None

        # Track keys
        if policy not in self.policy_samples:
            self.policy_samples[policy] = set()
        self.policy_samples[policy].add(sample)

        target_key = f"{policy}_{sample}"
        if target_key not in self.accumulated_data:
            self.accumulated_data[target_key] = {}
            self.day_data[target_key] = {}

        # Update metrics
        for k, v in record.items():
            if k in ["policy", "sample", "day", "timestamp", "routes", "total_fill"]:
                continue

            if k not in self.accumulated_data[target_key]:
                self.accumulated_data[target_key][k] = []

            self.accumulated_data[target_key][k].append(v)
            self.metrics.add(k)

        # Store rich data per day for bars and maps
        if "routes" in record or "total_fill" in record:
            self.day_data[target_key][day] = {
                "routes": record.get("routes"),
                "total_fill": record.get("total_fill"),
                "metrics": {k: v for k, v in record.items() if k in self.metrics},
            }

        return target_key, policy, sample

    def get_data(self, policy: str, sample: str, metric: str) -> List[Any]:
        """Retrieve accumulated time-series data."""
        key = f"{policy}_{sample}"
        return self.accumulated_data.get(key, {}).get(metric, [])

    def get_day_details(self, policy: str, sample: str, day: int) -> Optional[Dict[str, Any]]:
        """Retrieve rich data for a specific day."""
        key = f"{policy}_{sample}"
        return self.day_data.get(key, {}).get(day)
