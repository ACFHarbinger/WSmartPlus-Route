"""
Model enum for WSmart-Route.

Attributes:
    ModelTag: Enum for model tags

Example:
    >>> from logic.src.enums import ModelTag
    >>> ModelTag.TRANSFORMER
    <ModelTag.TRANSFORMER: 1>
"""

from enum import Enum, auto


class ModelTag(Enum):
    """
    Model tags for WSmart-Route.

    Attributes:
        TRANSFORMER: Transformer architecture
        GRAPH_NEURAL_NETWORK: Graph neural network architecture
        POINTER_NETWORK: Pointer network architecture
        MLP: Multi-layer perceptron architecture
        AUTOREGRESSIVE: Autoregressive decoding paradigm
        NON_AUTOREGRESSIVE: Non-autoregressive decoding paradigm
        ENCODER: Encoder component
        DECODER: Decoder component
        ACTOR: Actor component
        CRITIC: Critic component
        SYMMETRY_AWARE: Symmetry-aware model
        SUPPORTS_AUGMENTATION: Model supports augmentation
    """

    # Architecture
    TRANSFORMER = auto()
    GRAPH_NEURAL_NETWORK = auto()
    POINTER_NETWORK = auto()
    MLP = auto()

    # Decoding Paradigm
    AUTOREGRESSIVE = auto()  # Predicts node-by-node (step-by-step)
    NON_AUTOREGRESSIVE = auto()  # Predicts a full edge-heatmap in one shot (NAR)

    # Component Role
    ENCODER = auto()
    DECODER = auto()
    ACTOR = auto()
    CRITIC = auto()

    # Symmetry & Augmentation
    SYMMETRY_AWARE = auto()  # Invariant to rotation/translation (e.g., SymNCO)
    SUPPORTS_AUGMENTATION = auto()  # Supports 8x POMO augmentations
