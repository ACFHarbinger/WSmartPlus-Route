# {py:mod}`src.models.subnets.encoders.nargnn.gnn_layer`

```{py:module} src.models.subnets.encoders.nargnn.gnn_layer
```

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.gnn_layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GNNLayer <src.models.subnets.encoders.nargnn.gnn_layer.GNNLayer>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.nargnn.gnn_layer.GNNLayer
    :summary:
    ```
````

### API

`````{py:class} GNNLayer(embed_dim: int, act_fn: str = 'silu', agg_fn: str = 'mean')
:canonical: src.models.subnets.encoders.nargnn.gnn_layer.GNNLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.gnn_layer.GNNLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.gnn_layer.GNNLayer.__init__
```

````{py:method} forward(x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.encoders.nargnn.gnn_layer.GNNLayer.forward

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.gnn_layer.GNNLayer.forward
```

````

`````
