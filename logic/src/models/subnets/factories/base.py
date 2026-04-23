"""
Abstract base for neural component factories and decoder utility.

Attributes:
    NeuralComponentFactory: Abstract interface for creating encoders and decoders.
    _create_decoder_by_type: Internal utility for instantiating decoders by string identifiers.

Example:
    >>> # Factories are usually inherited or used via specific implementations
    >>> from logic.src.models.subnets.factories.base import NeuralComponentFactory
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.decoders.deepaco import ACODecoder
from logic.src.models.subnets.decoders.gat import DeepGATDecoder
from logic.src.models.subnets.decoders.glimpse.decoder import GlimpseDecoder
from logic.src.models.subnets.decoders.mdam import MDAMDecoder
from logic.src.models.subnets.decoders.polynet import PolyNetDecoder
from logic.src.models.subnets.decoders.ptr import PointerDecoder


def _create_decoder_by_type(decoder_type: str, **kwargs: Any) -> nn.Module:
    """Instantiates a decoder based on the specified type name.

    Args:
        decoder_type (str): Type identifier ('attention', 'deep', 'pointer', 'mdam', 'polynet', 'aco').
        kwargs: Configuration arguments passed directly to the decoder constructor.

    Returns:
        nn.Module: The instantiated decoder module.

    Raises:
        ValueError: If the decoder_type is not recognized.
    """
    decoder_type = decoder_type.lower()
    if decoder_type in ("attention", "glimpse"):
        return GlimpseDecoder(**kwargs)
    elif decoder_type in ("deep", "gat", "graph_attention"):
        return DeepGATDecoder(**kwargs)
    elif decoder_type == "pointer":
        return PointerDecoder(**kwargs)
    elif decoder_type == "mdam":
        return MDAMDecoder(**kwargs)
    elif decoder_type == "polynet":
        return PolyNetDecoder(**kwargs)
    elif decoder_type == "aco":
        return ACODecoder(**kwargs)
    else:
        raise ValueError(
            f"Unknown decoder_type: {decoder_type}. Choose from 'attention', 'glimpse', 'deep', 'pointer', 'mdam', 'polynet', 'aco'."
        )


class NeuralComponentFactory(ABC):
    """
    Abstract Factory for creating neural components (Encoders and Decoders).

    Attributes:
        None: Abstract base class does not maintain state.
    """

    @abstractmethod
    def create_encoder(
        self,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Create an encoder instance.

        Args:
            norm_config (Optional[NormalizationConfig]): Normalization settings. Defaults to None.
            activation_config (Optional[ActivationConfig]): Activation settings. Defaults to None.
            kwargs: Architecture-specific configuration.

        Returns:
            nn.Module: The instantiated encoder module.
        """
        pass

    @abstractmethod
    def create_decoder(
        self,
        decoder_type: str = "attention",
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Create a decoder instance based on decoder_type.

        Args:
            decoder_type (str): Type of decoder ('attention', 'deep', 'pointer'). Defaults to 'attention'.
            norm_config (Optional[NormalizationConfig]): Normalization settings. Defaults to None.
            activation_config (Optional[ActivationConfig]): Activation settings. Defaults to None.
            kwargs: Architecture-specific configuration.

        Returns:
            nn.Module: The instantiated decoder module.
        """
        pass
