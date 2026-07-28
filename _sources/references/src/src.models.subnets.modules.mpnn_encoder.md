# {py:mod}`src.models.subnets.modules.mpnn_encoder`

```{py:module} src.models.subnets.modules.mpnn_encoder
```

```{autodoc2-docstring} src.models.subnets.modules.mpnn_encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MPNNEncoder <src.models.subnets.modules.mpnn_encoder.MPNNEncoder>`
  - ```{autodoc2-docstring} src.models.subnets.modules.mpnn_encoder.MPNNEncoder
    :summary:
    ```
````

### API

`````{py:class} MPNNEncoder(num_layers: int, node_dim: int, edge_dim: int, hidden_dim: int = 64, aggr: str = 'add', norm: str = 'batch')
:canonical: src.models.subnets.modules.mpnn_encoder.MPNNEncoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.modules.mpnn_encoder.MPNNEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.mpnn_encoder.MPNNEncoder.__init__
```

````{py:method} forward(x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.modules.mpnn_encoder.MPNNEncoder.forward

```{autodoc2-docstring} src.models.subnets.modules.mpnn_encoder.MPNNEncoder.forward
```

````

`````
