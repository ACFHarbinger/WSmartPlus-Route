# {py:mod}`src.models.subnets.embeddings.ptp`

```{py:module} src.models.subnets.embeddings.ptp
```

```{autodoc2-docstring} src.models.subnets.embeddings.ptp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PTPInitEmbedding <src.models.subnets.embeddings.ptp.PTPInitEmbedding>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.ptp.PTPInitEmbedding
    :summary:
    ```
````

### API

`````{py:class} PTPInitEmbedding(embed_dim: int = 128, node_dim: int = 3, temporal_horizon: int = 0, legacy_depot_projection: bool = False)
:canonical: src.models.subnets.embeddings.ptp.PTPInitEmbedding

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.embeddings.ptp.PTPInitEmbedding
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.embeddings.ptp.PTPInitEmbedding.__init__
```

````{py:method} forward(td: typing.Union[tensordict.TensorDict, typing.Dict[str, typing.Any]], temporal_features: bool = True) -> torch.Tensor
:canonical: src.models.subnets.embeddings.ptp.PTPInitEmbedding.forward

```{autodoc2-docstring} src.models.subnets.embeddings.ptp.PTPInitEmbedding.forward
```

````

````{py:method} init_node_embeddings(nodes: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.models.subnets.embeddings.ptp.PTPInitEmbedding.init_node_embeddings

```{autodoc2-docstring} src.models.subnets.embeddings.ptp.PTPInitEmbedding.init_node_embeddings
```

````

`````
