# {py:mod}`src.models.subnets.embeddings.cvrpp`

```{py:module} src.models.subnets.embeddings.cvrpp
```

```{autodoc2-docstring} src.models.subnets.embeddings.cvrpp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CVRPPInitEmbedding <src.models.subnets.embeddings.cvrpp.CVRPPInitEmbedding>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.cvrpp.CVRPPInitEmbedding
    :summary:
    ```
````

### API

`````{py:class} CVRPPInitEmbedding(embed_dim: int = 128, node_dim: int = 3, temporal_horizon: int = 0, legacy_depot_projection: bool = False)
:canonical: src.models.subnets.embeddings.cvrpp.CVRPPInitEmbedding

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.embeddings.cvrpp.CVRPPInitEmbedding
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.embeddings.cvrpp.CVRPPInitEmbedding.__init__
```

````{py:method} forward(td: typing.Union[tensordict.TensorDict, typing.Dict[str, typing.Any]], temporal_features: bool = True) -> torch.Tensor
:canonical: src.models.subnets.embeddings.cvrpp.CVRPPInitEmbedding.forward

```{autodoc2-docstring} src.models.subnets.embeddings.cvrpp.CVRPPInitEmbedding.forward
```

````

````{py:method} init_node_embeddings(nodes: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.models.subnets.embeddings.cvrpp.CVRPPInitEmbedding.init_node_embeddings

```{autodoc2-docstring} src.models.subnets.embeddings.cvrpp.CVRPPInitEmbedding.init_node_embeddings
```

````

`````
