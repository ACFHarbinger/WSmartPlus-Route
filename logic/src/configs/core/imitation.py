"""Imitation specific configuration."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ImitationConfig:
    """Imitation specific configuration."""

    mode: str = "hgs"  # hgs, alns, random_ls, 2opt
    random_ls_iterations: int = 100
    random_ls_op_probs: Optional[Dict[str, float]] = None
    enabled: bool = False
    loss_fn: str = "nll"
