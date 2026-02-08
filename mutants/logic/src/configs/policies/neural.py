"""
Neural policy configuration.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class NeuralConfig:
    """Configuration for Neural Agent policy.

    Attributes:
        model_path: Path to the trained model weights.
        decode_type: Decoding strategy ('greedy', 'sampling', 'beam_search').
        temperature: Softmax temperature for sampling.
        beam_width: Beam width for beam search decoding.
        must_go: List of must-go strategy config files.
        post_processing: List of post-processing operations to apply.
    """

    model_path: Optional[str] = None
    decode_type: str = "greedy"
    temperature: float = 1.0
    beam_width: int = 1
    must_go: Optional[List[str]] = None
    post_processing: Optional[List[str]] = None
