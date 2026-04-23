"""Activation Function module.

This module provides the ActivationFunction wrapper, which simplifies the
instantiation of various PyTorch nonlinearities through a unified interface.

Attributes:
    ActivationFunction: Configurable activation layer factory.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.modules.activation_function import ActivationFunction
    >>> act = ActivationFunction(af_name="relu")
    >>> x = torch.randn(1, 10)
    >>> out = act(x)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig


class ActivationFunction(nn.Module):
    """Wrapper for various activation functions in PyTorch.

    Provides a unified interface to instantiate different activation functions
    by name, with support for parameters like threshold, negative slope, and
    learnable parameters.

    Attributes:
        activation (nn.Module): The underlying PyTorch activation module.
        tval (float): Stored threshold value for manual clipping.
        rval (float): Stored replacement value for manual clipping.
        apply_threshold (bool): Whether manual threshold clipping is active.
    """

    def __init__(
        self,
        af_name: Optional[str] = None,
        fparam: Optional[float] = None,
        tval: Optional[float] = None,
        rval: Optional[float] = None,
        n_params: Optional[int] = None,
        urange: Optional[Tuple[float, float]] = None,
        inplace: Optional[bool] = False,
        activation_config: Optional[ActivationConfig] = None,
    ) -> None:
        """Initializes ActivationFunction.

        Args:
            af_name: Name of the activation function (e.g., 'relu', 'leakyrelu').
            fparam: Primary float parameter (e.g., negative_slope, alpha).
            tval: Threshold value for shrinking or thresholding.
            rval: Replacement value for threshold-clipping.
            n_params: Parameter count for multi-channel activations (e.g., PReLU).
            urange: Optional range for Hardtanh or RReLU.
            inplace: Whether to perform the operation in-place.
            activation_config: Optional configuration object to override parameters.
        """
        super().__init__()

        # Use activation_config if provided, otherwise create from individual args
        if activation_config is None:
            activation_config = ActivationConfig(
                name=af_name if af_name is not None else "relu",
                param=fparam if fparam is not None else 1.0,
                threshold=tval if tval is not None else 6.0,
                replacement_value=rval if rval is not None else 6.0,
                n_params=n_params if n_params is not None else 3,
                range=list(urange) if urange is not None else [0.125, 1 / 3],
            )

        af_name = activation_config.name
        fparam = activation_config.param
        tval = activation_config.threshold
        rval = activation_config.replacement_value
        n_params = activation_config.n_params
        urange_tuple = (
            (activation_config.range[0], activation_config.range[1])
            if activation_config.range and len(activation_config.range) >= 2
            else None
        )

        if tval and rval is None and af_name != "softplus":
            rval = tval  # Replacement value = threshold

        self.activation = self._get_activation_module(af_name, fparam, tval, rval, n_params, urange_tuple, inplace)

        if isinstance(self.activation, nn.PReLU):
            self.init_parameters()

        if tval and rval and not isinstance(self.activation, nn.Threshold):
            self.tval = tval
            self.rval = rval
            self.apply_threshold = True
        else:
            self.apply_threshold = False

    def _get_activation_module(
        self,
        af_name: str,
        fparam: Optional[float],
        tval: Optional[float],
        rval: Optional[float],
        n_params: Optional[int],
        urange: Optional[Tuple[float, float]],
        inplace: Optional[bool],
    ) -> nn.Module:
        """Factory method to create PyTorch activation modules.

        Args:
            af_name: Requested activation name.
            fparam: Parameter (alpha, negative_slope, etc.).
            tval: Threshold lambd or value.
            rval: Replacement value.
            n_params: Parameter count for PReLU.
            urange: Range tuple for stochastic/bounded acts.
            inplace: In-place flag.

        Returns:
            nn.Module: Instantiated PyTorch activation.

        Raises:
            ValueError: If the activation name is unknown.
        """
        # Simple mapping for one-to-one activations
        inplace_val = bool(inplace)
        simple_mappings = {
            "relu": lambda: nn.ReLU(inplace=inplace_val),
            "leakyrelu": lambda: nn.LeakyReLU(
                inplace=inplace_val,
                negative_slope=fparam if fparam is not None else 1e-2,
            ),
            "silu": lambda: nn.SiLU(inplace=inplace_val),
            "selu": lambda: nn.SELU(inplace=inplace_val),
            "elu": lambda: nn.ELU(inplace=inplace_val, alpha=fparam if fparam is not None else 1.0),
            "celu": lambda: nn.CELU(inplace=inplace_val, alpha=fparam if fparam is not None else 1.0),
            "gelu": lambda: nn.GELU(),
            "gelu_tanh": lambda: nn.GELU(approximate="tanh"),
            "prelu": lambda: nn.PReLU(num_parameters=n_params if n_params else 1, init=fparam if fparam else 0.25),
            "rrelu": lambda: nn.RReLU(
                inplace=inplace_val,
                lower=urange[0] if urange else 1.0 / 8,
                upper=urange[1] if urange else 1.0 / 3,
            ),
        }

        if af_name in simple_mappings:
            return simple_mappings[af_name]()

        # Handle capitalize-based mappings
        caps_list = [
            "tanh",
            "tanhshrink",
            "sigmoid",
            "logsigmoid",
            "softsign",
            "mish",
            "hardswish",
            "hardsigmoid",
        ]
        if af_name in caps_list:
            cls_name = af_name.capitalize() if af_name != "logsigmoid" else "LogSigmoid"
            if af_name == "tanhshrink":
                cls_name = "Tanhshrink"
            elif af_name == "hardswish":
                cls_name = "Hardswish"
            elif af_name == "hardsigmoid":
                cls_name = "Hardsigmoid"

            cls = getattr(nn, cls_name)
            return cls(inplace=inplace_val) if hasattr(cls, "inplace") else cls()

        # Handle other parameterized activations
        if af_name == "hardshrink":
            return nn.Hardshrink(lambd=fparam if fparam is not None else 0.5)
        if af_name == "hardtanh":
            return nn.Hardtanh(
                inplace=inplace_val,
                min_val=urange[0] if urange else -1.0,
                max_val=urange[1] if urange else 1.0,
            )
        if af_name == "glu":
            return nn.GLU(dim=int(fparam) if fparam is not None else -1)
        if af_name == "threshold":
            return nn.Threshold(
                inplace=inplace_val,
                threshold=tval if tval is not None else 0.0,
                value=rval if rval is not None else 0.0,
            )
        if af_name == "softplus":
            return nn.Softplus(
                beta=int(fparam) if fparam is not None else 1,
                threshold=int(tval) if tval is not None else 20,
            )
        raise ValueError(f"Unknown activation function: {af_name}")

    def init_parameters(self) -> None:
        """Initializes internal parameters using uniform distribution."""
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Applies the activation function to the input tensor.

        Args:
            input: Unactivated input tensor.
            mask: Optional mask (not used by default activations).

        Returns:
            torch.Tensor: Activated output tensor.
        """
        out = self.activation(input)
        if self.apply_threshold:
            out = torch.where(out > self.tval, self.rval, out)
        return out
