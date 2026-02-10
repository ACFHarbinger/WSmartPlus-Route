"""Optimized Graph Convolution implementation with multiple aggregators."""

from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, SparseTensor, torch_sparse
from torch_geometric.utils import add_remaining_self_loops, scatter


# Adapted from https://github.com/shyam196/egc
class EfficientGraphConvolution(MessagePassing):
    """
    Efficient Graph Convolution (EGC) with multiple aggregators.

    This layer computes node updates using a linear combination of different
    neighborhood aggregations (mean, max, sum, var, std, symnorm) and self-features.
    Supports multi-head weights and basis functions for efficiency.
    """

    _cached_edge_index: Optional[Tuple[Tensor, OptTensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggrs: Iterable[str] = ("symnorm",),
        n_heads: int = 8,
        num_bases: int = 4,
        cached: bool = False,
        add_self_loops: bool = True,
        bias: bool = True,
        sigmoid: bool = False,
        **kwargs,
    ):
        """
        Args:
            in_channels: Dimension of input features.
            out_channels: Dimension of output features.
            aggrs: Iterable of aggregator names to use (e.g., "sum", "mean", "symnorm").
            n_heads: Number of attention heads.
            num_bases: Number of basis functions for the weight matrix.
            cached: If set to `True`, the layer will cache the computation of
                :obj:`edge_index` and :obj:`symnorm_weight` on first execution,
                and will use the cached values for further executions.
            add_self_loops: If set to `False`, will not add self-loops to the
                input graph.
            bias: Whether to use a bias term.
            sigmoid: If set to `True`, applies a sigmoid activation to the weighting coefficients.
        """
        super(EfficientGraphConvolution, self).__init__(node_dim=1, **kwargs)
        if out_channels % n_heads != 0:
            raise ValueError("out_channels must be divisible by the number of heads")

        for a in aggrs:
            if a not in {"sum", "mean", "symnorm", "min", "max", "var", "std"}:
                raise ValueError("Unsupported aggregator: {}".format(a))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.num_bases = num_bases
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.aggregators = tuple(aggrs)  # Convert to tuple for mypy compatibility
        self.sigmoid = sigmoid

        self.bases_weight = Parameter(torch.Tensor(in_channels, (out_channels // n_heads) * num_bases))
        self.comb_weight = Linear(in_channels, n_heads * num_bases * len(self.aggregators))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters of the layer using Glorot initialization."""
        glorot(self.bases_weight)
        self.comb_weight.reset_parameters()
        zeros(self.bias)
        self._cached_adj_t = None
        self._cached_edge_index = None

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """
        Forward pass for Efficient Graph Convolution.

        Args:
            x: Node features tensor of shape (batch_size, num_nodes, in_channels).
            edge_index: Graph adjacency information.

        Returns:
            Updated node features tensor.
        """
        symnorm_weight: OptTensor = None
        if "symnorm" in self.aggregators or self.add_self_loops:
            edge_index, symnorm_weight = self._norm_and_cache(x, edge_index)

        batch_size = x.size(0)
        num_nodes = x.size(1)

        # [num_nodes, (out_channels // n_heads) * num_bases]
        bases = torch.matmul(x, self.bases_weight)
        # [num_nodes, n_heads * num_bases * num_aggrs]
        weightings = self.comb_weight(x)
        if self.sigmoid:
            weightings = torch.sigmoid_(weightings)

        if symnorm_weight is not None:
            symnorm_weight = symnorm_weight.view(-1, 1)

        # [num_nodes, num_aggregators, (out_channels // n_heads) * num_bases]
        # propagate_type: (x: Tensor, symnorm_weight: OptTensor)
        aggregated = self.propagate(edge_index, x=bases, symnorm_weight=symnorm_weight, size=None)

        weightings = weightings.view(
            batch_size,
            num_nodes,
            self.n_heads,
            self.num_bases * len(self.aggregators),
        )
        aggregated = aggregated.view(
            batch_size,
            num_nodes,
            len(self.aggregators) * self.num_bases,
            self.out_channels // self.n_heads,
        )

        # [num_nodes, n_heads, out_channels // n_heads]
        out = torch.matmul(weightings, aggregated)
        out = out.view(batch_size, num_nodes, self.out_channels)
        if self.bias is not None:
            out += self.bias

        return out

    def _norm_and_cache(self, x: Tensor, edge_index: Adj) -> Tuple[Adj, OptTensor]:
        """Helper to handle graph normalization and caching."""
        num_nodes = x.size(self.node_dim)

        if "symnorm" in self.aggregators:
            if isinstance(edge_index, Tensor):
                if self._cached_edge_index is None:
                    edge_index, sw = gcn_norm(edge_index, num_nodes=num_nodes, add_self_loops=self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, sw)

                    return edge_index, sw
                return self._cached_edge_index

            if self._cached_adj_t is None:
                adj_t = gcn_norm(edge_index, num_nodes=num_nodes, add_self_loops=self.add_self_loops)
                if self.cached:
                    self._cached_adj_t = adj_t

                return adj_t, None
            return self._cached_adj_t, None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                if self._cached_edge_index is None:
                    edge_index, _ = add_remaining_self_loops(edge_index)
                    if self.cached:
                        self._cached_edge_index = (edge_index, None)

                    return edge_index, None
                return self._cached_edge_index[0], None

            if self._cached_adj_t is None:
                adj_t = torch_sparse.fill_diag(edge_index, 1.0)
                if self.cached:
                    self._cached_adj_t = adj_t

                return adj_t, None
            return self._cached_adj_t, None

        return edge_index, None

    def message(self, x_j: Tensor) -> Tensor:
        """
        Passes messages along edges.
        """
        return x_j

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        dim_size: Optional[int] = None,
        symnorm_weight: OptTensor = None,
    ) -> Tensor:
        """Aggregates messages from neighbors using multiple aggregators."""
        aggregated = []
        inputs = inputs.permute(1, 0, 2)
        for aggregator in self.aggregators:
            out = self._run_aggregator(aggregator, inputs, index, dim_size, symnorm_weight)
            aggregated.append(out)
        return torch.stack(aggregated, dim=1)

    def _run_aggregator(
        self,
        aggregator: str,
        inputs: Tensor,
        index: Tensor,
        dim_size: Optional[int],
        symnorm_weight: OptTensor,
    ) -> Tensor:
        """Execute a single neighborhood aggregator."""
        if aggregator == "sum":
            return scatter(inputs, index, 0, dim_size, reduce="sum")
        if aggregator == "symnorm":
            assert symnorm_weight is not None
            return scatter(inputs * symnorm_weight.unsqueeze(-1), index, 0, dim_size, reduce="sum")
        if aggregator == "mean":
            return scatter(inputs, index, 0, dim_size, reduce="mean")
        if aggregator == "min":
            return scatter(inputs, index, 0, dim_size, reduce="min")
        if aggregator == "max":
            return scatter(inputs, index, 0, dim_size, reduce="max")
        if aggregator in {"var", "std"}:
            mean = scatter(inputs, index, 0, dim_size, reduce="mean")
            mean_squares = scatter(inputs * inputs, index, 0, dim_size, reduce="mean")
            out = mean_squares - mean * mean
            if aggregator == "std":
                out = torch.sqrt(torch.relu(out) + 1e-5)
            return out
        raise ValueError(f'Unknown aggregator "{aggregator}".')

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        """
        Performs message passing and aggregation in a single step for sparse tensors.
        """
        aggregated = []
        adj_t_nonorm = adj_t.set_value(None) if len(self.aggregators) > 1 and "symnorm" in self.aggregators else adj_t

        for aggregator in self.aggregators:
            out = self._run_sparse_aggregator(aggregator, adj_t, adj_t_nonorm, x)
            aggregated.append(out)

        return torch.stack(aggregated, dim=1)

    def _run_sparse_aggregator(
        self, aggregator: str, adj_t: SparseTensor, adj_t_nonorm: SparseTensor, x: Tensor
    ) -> Tensor:
        """Execute a single sparse neighborhood aggregator."""
        if aggregator == "symnorm":
            return torch_sparse.matmul(adj_t, x, reduce="sum")

        if aggregator in ["var", "std"]:
            mean = torch_sparse.matmul(adj_t_nonorm, x, reduce="mean")
            mean_sq = torch_sparse.matmul(adj_t_nonorm, x * x, reduce="mean")
            out = mean_sq - mean * mean
            if aggregator == "std":
                out = torch.sqrt(torch.relu(out) + 1e-5)
            return out

        return torch_sparse.matmul(adj_t_nonorm, x, reduce=aggregator)

    def __repr__(self):
        """String representation of the layer."""
        return "{}({}, {}, {})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.aggregators,
        )
