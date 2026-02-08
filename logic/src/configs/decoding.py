"""
Decoding Config module.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DecodingConfig:
    """Configuration for decoding strategies."""

    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    tanh_clipping: float = 0.0
    mask_logits: bool = True
    multistart: bool = False
    num_starts: int = 1
    select_best: bool = False
