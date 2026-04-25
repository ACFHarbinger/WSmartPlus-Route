"""
Decoding Strategy Config module.

Attributes:
    DecodingConfig: Decoding strategy configuration.

Example:
    >>> from logic.src.configs.models import DecodingConfig
    >>> config = DecodingConfig()
    >>> print(config)
    DecodingConfig(strategy='greedy', beam_width=1, temperature=1.0, top_k=None, top_p=None, tanh_clipping=0.0, mask_logits=True, multistart=False, num_starts=1, select_best=False)
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DecodingConfig:
    """Configuration for decoding strategies.

    Attributes:
        strategy (str): Decoding method.
        beam_width (Any): Beam search width or sampling size.
        temperature (float): Temperature for sampling.
        top_k (Optional[int]): Top-k sampling.
        top_p (Optional[float]): Top-p sampling.
        tanh_clipping (float): Tanh clipping value.
        mask_logits (bool): Masking of logits.
        multistart (bool): Multistart decoding.
        num_starts (int): Number of starts for multistart decoding.
        select_best (bool): Selection of the best decoding.
    """

    strategy: str = "greedy"  # greedy, sampling, beam_search
    beam_width: Any = 1  # For beam search or sampling size
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    tanh_clipping: float = 0.0
    mask_logits: bool = True
    multistart: bool = False
    num_starts: int = 1
    select_best: bool = False
