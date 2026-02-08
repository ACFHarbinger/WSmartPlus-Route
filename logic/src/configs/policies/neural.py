"""
Neural policy configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from ..envs.objective import ObjectiveConfig
from ..models.decoding import DecodingConfig
from ..models.model import ModelConfig
from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class NeuralConfig:
    """Configuration for Neural Agent policy.

    Attributes:
        model: Model configuration (architecture and load path).
        decoding: Decoding strategy configuration.
        reward: Objective/reward weights configuration.
        must_go: List of must-go strategy config files.
        post_processing: List of post-processing operations to apply.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    reward: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
