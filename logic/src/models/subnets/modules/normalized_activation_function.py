"""
Normalized activation functions (Softmax, etc.) with adaptive options.

Attributes:
    NormalizedActivationFunction: Wrapper for diverse normalized activation layers.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.modules.normalized_activation_function import NormalizedActivationFunction
    >>> act = NormalizedActivationFunction(naf_name="softmax", dim=-1)
    >>> x = torch.randn(1, 10)
    >>> out = act(x)
"""

import math
from typing import Optional, Sequence

import torch
from torch import nn


class NormalizedActivationFunction(nn.Module):
    """
    Wrapper for normalized activation functions (Softmax, LogSoftmax, etc.).

    Provides a unified interface for standard PyTorch normalized activations
    and more complex ones like AdaptiveLogSoftmax.

    Attributes:
        norm_activation (nn.Module): The underlying normalization layer.
    """

    def __init__(
        self,
        naf_name: str = "softmax",
        dim: Optional[int] = -1,
        n_classes: Optional[int] = None,
        cutoffs: Optional[Sequence[int]] = None,
        dval: Optional[float] = 4.0,
        bias: Optional[bool] = False,
    ):
        """
        Initializes the normalized activation function.

        Args:
            naf_name (str): Name of the normalized activation function
                ('softmax', 'logsoftmax', 'softmin', 'softmax2d', 'adaptivelogsoftmax').
            dim (Optional[int]): Dimension along which the activation is applied. Defaults to -1.
            n_classes (Optional[int]): Number of classes (required for adaptive softmax).
            cutoffs (Optional[Sequence[int]]): Cutoffs for adaptive softmax clusters.
            dval (Optional[float]): Divisor value for adaptive softmax. Defaults to 4.0.
            bias (Optional[bool]): Whether to insert bias in adaptive softmax. Defaults to False.

        Raises:
            ValueError: If unknown activation name or missing required parameters.
        """
        super(NormalizedActivationFunction, self).__init__()
        self.norm_activation: nn.Module
        if naf_name == "softmin":
            self.norm_activation = nn.Softmin(dim=dim)
        elif naf_name == "softmax":
            self.norm_activation = nn.Softmax(dim=dim)
        elif naf_name == "logsoftmax":
            self.norm_activation = nn.LogSoftmax(dim=dim)
        elif naf_name == "softmax2d":
            self.norm_activation = nn.Softmax2d()
        elif naf_name == "adaptivelogsoftmax":
            if n_classes is None:
                raise ValueError("n_classes must be provided for adaptivelogsoftmax")
            # Ensure cutoffs is list
            if cutoffs is None:
                cutoffs = [n_classes // 4, n_classes // 2, 3 * n_classes // 4]
            self.norm_activation = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=dim if dim is not None else 0,
                n_classes=n_classes,
                cutoffs=cutoffs,
                div_value=float(dval) if dval is not None else 4.0,
                head_bias=bool(bias) if bias is not None else False,
            )
        else:
            raise ValueError("Unknown normalized activation function: {}".format(naf_name))

        if isinstance(self.norm_activation, nn.AdaptiveLogSoftmaxWithLoss):
            self.init_parameters()

    def init_parameters(self) -> None:
        """Initializes the parameters using Xavier-like uniform distribution.

        Only applies if the underlying activation has learnable parameters
        (e.g., AdaptiveLogSoftmax).
        """
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies the normalized activation function to the input.

        Args:
            input (torch.Tensor): Input tensor.
            mask (Optional[torch.Tensor]): Optional mask or target (used by AdaptiveLogSoftmax).

        Returns:
            torch.Tensor: Normalized output or log probabilities.
        """
        if isinstance(self.norm_activation, nn.AdaptiveLogSoftmaxWithLoss):
            # If mask is provided, treat it as the target for loss calculation
            if mask is not None:
                return self.norm_activation(input, mask)
            # Otherwise, return the log probabilities (inference behavior)
            return self.norm_activation.log_prob(input)
        return self.norm_activation(input)
