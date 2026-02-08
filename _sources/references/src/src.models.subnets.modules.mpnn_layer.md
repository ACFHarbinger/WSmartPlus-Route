# {py:mod}`src.models.subnets.modules.mpnn_layer`

```{py:module} src.models.subnets.modules.mpnn_layer
```

```{autodoc2-docstring} src.models.subnets.modules.mpnn_layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MessagePassingLayer <src.models.subnets.modules.mpnn_layer.MessagePassingLayer>`
  - ```{autodoc2-docstring} src.models.subnets.modules.mpnn_layer.MessagePassingLayer
    :summary:
    ```
````

### API

`````{py:class} MessagePassingLayer(node_dim: int, edge_dim: int, hidden_dim: int = 64, aggr: str = 'add', norm: str = 'batch', bias: bool = True)
:canonical: src.models.subnets.modules.mpnn_layer.MessagePassingLayer

Bases: {py:obj}`torch_geometric.nn.MessagePassing`

```{autodoc2-docstring} src.models.subnets.modules.mpnn_layer.MessagePassingLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.mpnn_layer.MessagePassingLayer.__init__
```

````{py:method} forward(x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.modules.mpnn_layer.MessagePassingLayer.forward

```{autodoc2-docstring} src.models.subnets.modules.mpnn_layer.MessagePassingLayer.forward
```

````

````{py:method} message(x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.modules.mpnn_layer.MessagePassingLayer.message

```{autodoc2-docstring} src.models.subnets.modules.mpnn_layer.MessagePassingLayer.message
```

````

````{py:method} update(aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.modules.mpnn_layer.MessagePassingLayer.update

```{autodoc2-docstring} src.models.subnets.modules.mpnn_layer.MessagePassingLayer.update
```

````

`````
