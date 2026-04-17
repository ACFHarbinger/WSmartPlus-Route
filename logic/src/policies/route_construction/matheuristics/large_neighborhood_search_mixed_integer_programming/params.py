from dataclasses import dataclass, fields
from typing import Any


@dataclass
class LNSMIPParams:
    k_destroy: int = 10
    d_destroy: int = 3
    max_iterations: int = 200
    mip_time_limit: float = 10.0
    mip_gap: float = 0.01
    acceptance: str = "improving"  # "improving" | "sa"
    sa_temperature: float = 100.0
    sa_cooling: float = 0.97
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> "LNSMIPParams":
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})
        return cls(**{f.name: getattr(config, f.name, f.default) for f in fields(cls)})
