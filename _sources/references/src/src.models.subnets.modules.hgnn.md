# {py:mod}`src.models.subnets.modules.hgnn`

```{py:module} src.models.subnets.modules.hgnn
```

```{autodoc2-docstring} src.models.subnets.modules.hgnn
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HetGNNLayer <src.models.subnets.modules.hgnn.HetGNNLayer>`
  - ```{autodoc2-docstring} src.models.subnets.modules.hgnn.HetGNNLayer
    :summary:
    ```
````

### API

`````{py:class} HetGNNLayer(node_types: list[str], edge_types: list[tuple[str, str, str]], hidden_dim: int = 64, aggr: str = 'sum', norm: str = 'layer')
:canonical: src.models.subnets.modules.hgnn.HetGNNLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.modules.hgnn.HetGNNLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.hgnn.HetGNNLayer.__init__
```

````{py:method} forward(x_dict: typing.Dict[str, torch.Tensor], edge_index_dict: typing.Dict[typing.Tuple[str, str, str], torch.Tensor]) -> typing.Dict[str, torch.Tensor]
:canonical: src.models.subnets.modules.hgnn.HetGNNLayer.forward

```{autodoc2-docstring} src.models.subnets.modules.hgnn.HetGNNLayer.forward
```

````

`````
