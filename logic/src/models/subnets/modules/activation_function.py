import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from logic.src.configs.models.activation_function import ActivationConfig


class ActivationFunction(nn.Module):
    """
    Wrapper for various activation functions in PyTorch.

    Provides a unified interface to instantiate different activation functions
    by name, with support for parameters like threshold, negative slope, etc.
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
    ):
        """
        Initializes the activation function.

        Args:
            af_name: Name of the activation function (e.g., 'relu', 'leakyrelu').
            fparam: Float parameter for activations (e.g., alpha for ELU, negative_slope for LeakyReLU).
            tval: Threshold value (used in Softshrink, Hardshrink, etc.).
            rval: Replacement value (used in global replacement logic if needed).
            n_params: Number of parameters for PReLU.
            urange: Uniform range for RReLU (lower, upper).
            inplace: Whether to perform the operation in-place.
            activation_config: Activation function configuration object.
        """
        super(ActivationFunction, self).__init__()

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
        urange = (
            (activation_config.range[0], activation_config.range[1])
            if activation_config.range and len(activation_config.range) >= 2
            else None
        )
        self.activation: nn.Module
        if tval and rval is None and not af_name == "softplus":
            rval = tval  # Replacement value = threshold
        if af_name == "relu":
            self.activation = nn.ReLU(inplace=bool(inplace))
        elif af_name == "leakyrelu":
            self.activation = nn.LeakyReLU(inplace=bool(inplace), negative_slope=fparam if fparam is not None else 1e-2)
        elif af_name == "silu":
            self.activation = nn.SiLU(inplace=bool(inplace))
        elif af_name == "selu":
            self.activation = nn.SELU(inplace=bool(inplace))
        elif af_name == "elu":
            self.activation = nn.ELU(inplace=bool(inplace), alpha=fparam if fparam is not None else 1.0)
        elif af_name == "celu":
            self.activation = nn.CELU(inplace=bool(inplace), alpha=fparam if fparam is not None else 1.0)
        elif af_name == "prelu":
            self.activation = nn.PReLU(
                num_parameters=n_params if n_params else 1,
                init=fparam if fparam else 0.25,
            )
        elif af_name == "rrelu":
            lower = urange[0] if urange else 1.0 / 8
            upper = urange[1] if urange else 1.0 / 3
            self.activation = nn.RReLU(inplace=bool(inplace), lower=lower, upper=upper)
        elif af_name == "gelu":
            self.activation = nn.GELU()
        elif af_name == "gelu_tanh":
            self.activation = nn.GELU(approximate="tanh")
        elif af_name == "tanh":
            self.activation = nn.Tanh()
        elif af_name == "tanhshrink":
            self.activation = nn.Tanhshrink()
        elif af_name == "mish":
            self.activation = nn.Mish(inplace=bool(inplace))
        elif af_name == "hardshrink":
            self.activation = nn.Hardshrink(lambd=fparam if fparam is not None else 0.5)
        elif af_name == "hardtanh":
            min_val = urange[0] if urange else -1.0
            max_val = urange[1] if urange else 1.0
            self.activation = nn.Hardtanh(inplace=bool(inplace), min_val=min_val, max_val=max_val)
        elif af_name == "hardswish":
            self.activation = nn.Hardswish(inplace=bool(inplace))
        elif af_name == "glu":
            self.activation = nn.GLU(dim=int(fparam) if fparam is not None else -1)
        elif af_name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif af_name == "logsigmoid":
            self.activation = nn.LogSigmoid()
        elif af_name == "hardsigmoid":
            self.activation = nn.Hardsigmoid(inplace=bool(inplace))
        elif af_name == "threshold":
            self.activation = nn.Threshold(
                inplace=bool(inplace),
                threshold=tval if tval is not None else 0.0,
                value=rval if rval is not None else 0.0,
            )
        elif af_name == "softplus":
            self.activation = nn.Softplus(
                beta=int(fparam) if fparam is not None else 1, threshold=int(tval) if tval is not None else 20
            )
        elif af_name == "softshrink":
            self.activation = nn.Softshrink(lambd=fparam if fparam is not None else 0.5)
        elif af_name == "softsign":
            self.activation = nn.Softsign()
        else:
            raise ValueError("Unknown activation function: {}".format(af_name))

        if isinstance(self.activation, nn.PReLU):
            self.init_parameters()

        if tval and rval and not isinstance(self.activation, nn.Threshold):
            self.tval = tval
            self.rval = rval
            self.apply_threshold = True
        else:
            self.apply_threshold = False

    def init_parameters(self):
        """Initializes the parameters of the activation function using uniform distribution."""
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):
        """
        Applies the activation function to the input.

        Args:
            input: Input tensor.
            mask: Optional mask (not used by default activations but kept for interface).

        Returns:
            Output tensor.
        """
        out = self.activation(input)
        if self.apply_threshold:
            out = torch.where(out > self.tval, self.rval, out)
        return out
