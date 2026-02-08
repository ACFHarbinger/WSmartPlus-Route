# {py:mod}`src.models.subnets.encoders.nargnn.gnn_encoder`

```{py:module} src.models.subnets.encoders.nargnn.gnn_encoder
```

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.gnn_encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimplifiedGNNEncoder <src.models.subnets.encoders.nargnn.gnn_encoder.SimplifiedGNNEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.nargnn.gnn_encoder.SimplifiedGNNEncoder
    :summary:
    ```
````

### API

`````{py:class} SimplifiedGNNEncoder(num_layers: int, embed_dim: int, act_fn: str = 'silu', agg_fn: str = 'mean')
:canonical: src.models.subnets.encoders.nargnn.gnn_encoder.SimplifiedGNNEncoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.gnn_encoder.SimplifiedGNNEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.gnn_encoder.SimplifiedGNNEncoder.__init__
```

````{py:method} forward(x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.encoders.nargnn.gnn_encoder.SimplifiedGNNEncoder.forward

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.gnn_encoder.SimplifiedGNNEncoder.forward
```

````

`````
