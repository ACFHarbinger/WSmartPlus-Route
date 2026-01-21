from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class Normalization(nn.Module):
    """
    Wrapper for various Normalization layers (Batch, Layer, Instance).
    """

    normalizer: nn.Module

    def __init__(
        self,
        embed_dim: int,
        norm_name: str = "batch",
        eps_alpha: float = 1e-05,
        learn_affine: Optional[bool] = True,
        track_stats: Optional[bool] = False,
        mbval: Optional[float] = 0.1,
        n_groups: Optional[int] = None,
        kval: Optional[float] = None,
        bias: Optional[bool] = True,
    ) -> None:
        """
        Initializes the normalization layer.

        Args:
            embed_dim: Embedding dimension.
            norm_name: Type of normalization ('batch', 'layer', 'instance', 'group', 'local_response').
            eps_alpha: Epsilon value for numerical stability.
            learn_affine: If True, learn affine parameters (weight and bias).
            track_stats: If True, track running statistics for BatchNorm/InstanceNorm.
            mbval: Momentum for running statistics or beta for LocalResponseNorm.
            n_groups: Number of groups for GroupNorm.
            kval: k value for LocalResponseNorm.
            bias: If True, add bias for LayerNorm.
        """
        super(Normalization, self).__init__()

        if norm_name == "instance":
            self.normalizer = nn.InstanceNorm1d(
                embed_dim,
                eps=eps_alpha,
                affine=learn_affine if learn_affine is not None else True,
                track_running_stats=track_stats if track_stats is not None else False,
                momentum=mbval if mbval is not None else 0.1,
            )
        elif norm_name == "batch":
            self.normalizer = nn.BatchNorm1d(
                embed_dim,
                eps=eps_alpha,
                affine=learn_affine if learn_affine is not None else True,
                track_running_stats=track_stats if track_stats is not None else False,
                momentum=mbval if mbval is not None else 0.1,
            )
        elif norm_name == "layer":
            self.normalizer = nn.LayerNorm(
                embed_dim,
                eps=eps_alpha,
                elementwise_affine=learn_affine if learn_affine is not None else True,
                bias=bias if bias is not None else True,
            )
        elif norm_name == "group":
            if n_groups is None:
                n_groups = 1
            self.normalizer = nn.GroupNorm(
                n_groups,
                eps=eps_alpha,
                num_channels=embed_dim,
                affine=learn_affine if learn_affine is not None else True,
            )
        elif norm_name == "local_response":
            self.normalizer = nn.LocalResponseNorm(
                embed_dim,
                alpha=eps_alpha,
                beta=mbval if mbval is not None else 0.75,
                k=kval if kval is not None else 1.0,
            )
        else:
            raise ValueError(f"Unknown normalization method: {norm_name}")

        if learn_affine:
            self.init_parameters()

    def init_parameters(self) -> None:
        """Initializes the affine parameters if applicable."""
        for param in self.parameters():
            if param.dim() > 0:
                stdv: float = 1.0 / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies the normalization to the input.

        Args:
            input: Input tensor.
            mask: Optional mask (currently not used by normalization layers).

        Returns:
            Normalized tensor.
        """
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        elif isinstance(self.normalizer, nn.LayerNorm):
            return self.normalizer(input)
        else:
            return self.normalizer(input)
