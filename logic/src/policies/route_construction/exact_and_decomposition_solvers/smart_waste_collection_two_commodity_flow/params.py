"""
Configuration parameters for the Smart Waste Collection - Two-Commodity Flow (SWC-TCF) policy.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class SWCTCFParams:
    """
    Configuration parameters for the SWC-TCF solver.

    Attributes:
        framework: Optimization framework to use ('ortools' or 'pyomo').
        engine: Optimization engine to use ('gurobi', 'scip', 'highs', or 'cplex').
        time_limit: Time limit for the solver in seconds.
        seed: Random seed for reproducibility.
    """

    framework: str = "ortools"
    engine: str = "gurobi"
    time_limit: float = 60.0
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> SWCTCFParams:
        """Create SWCTCFParams from a configuration object or dictionary.

        Performs explicit type casting for numeric fields to ensure consistency
        with the framework's configuration loading logic.
        """
        if config is None:
            return cls()

        raw_data: Dict[str, Any] = {}
        if isinstance(config, dict):
            raw_data = config
        else:
            for f in fields(cls):
                if hasattr(config, f.name):
                    raw_data[f.name] = getattr(config, f.name)

        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            val = raw_data.get(f.name, getattr(cls, f.name, f.default))
            if val is not None:
                if f.type is float or f.type == "float":
                    val = float(val)
                elif f.type is int or f.type == "int":
                    val = int(val)
            kwargs[f.name] = val

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Params to a dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
