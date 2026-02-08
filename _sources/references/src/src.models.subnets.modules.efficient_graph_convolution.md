# {py:mod}`src.models.subnets.modules.efficient_graph_convolution`

```{py:module} src.models.subnets.modules.efficient_graph_convolution
```

```{autodoc2-docstring} src.models.subnets.modules.efficient_graph_convolution
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EfficientGraphConvolution <src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution>`
  - ```{autodoc2-docstring} src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution
    :summary:
    ```
````

### API

`````{py:class} EfficientGraphConvolution(in_channels: int, out_channels: int, aggrs: typing.Iterable[str] = ('symnorm', ), n_heads: int = 8, num_bases: int = 4, cached: bool = False, add_self_loops: bool = True, bias: bool = True, sigmoid: bool = False, **kwargs)
:canonical: src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution

Bases: {py:obj}`torch_geometric.nn.MessagePassing`

```{autodoc2-docstring} src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution.__init__
```

````{py:attribute} _cached_edge_index
:canonical: src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution._cached_edge_index
:type: typing.Optional[typing.Tuple[torch.Tensor, torch_geometric.typing.OptTensor]]
:value: >
   None

```{autodoc2-docstring} src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution._cached_edge_index
```

````

````{py:attribute} _cached_adj_t
:canonical: src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution._cached_adj_t
:type: typing.Optional[torch_geometric.typing.SparseTensor]
:value: >
   None

```{autodoc2-docstring} src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution._cached_adj_t
```

````

````{py:method} reset_parameters()
:canonical: src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution.reset_parameters

```{autodoc2-docstring} src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution.reset_parameters
```

````

````{py:method} forward(x: torch.Tensor, edge_index: torch_geometric.typing.Adj) -> torch.Tensor
:canonical: src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution.forward

```{autodoc2-docstring} src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution.forward
```

````

````{py:method} message(x_j: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution.message

```{autodoc2-docstring} src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution.message
```

````

````{py:method} aggregate(inputs: torch.Tensor, index: torch.Tensor, dim_size: typing.Optional[int] = None, symnorm_weight: torch_geometric.typing.OptTensor = None) -> torch.Tensor
:canonical: src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution.aggregate

```{autodoc2-docstring} src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution.aggregate
```

````

````{py:method} message_and_aggregate(adj_t: torch_geometric.typing.SparseTensor, x: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution.message_and_aggregate

```{autodoc2-docstring} src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution.message_and_aggregate
```

````

````{py:method} __repr__()
:canonical: src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution.__repr__

```{autodoc2-docstring} src.models.subnets.modules.efficient_graph_convolution.EfficientGraphConvolution.__repr__
```

````

`````
