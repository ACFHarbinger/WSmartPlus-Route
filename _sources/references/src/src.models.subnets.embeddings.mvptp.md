# {py:mod}`src.models.subnets.embeddings.mvptp`

```{py:module} src.models.subnets.embeddings.mvptp
```

```{autodoc2-docstring} src.models.subnets.embeddings.mvptp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MVPTPInitEmbedding <src.models.subnets.embeddings.mvptp.MVPTPInitEmbedding>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.mvptp.MVPTPInitEmbedding
    :summary:
    ```
````

### API

`````{py:class} MVPTPInitEmbedding(embed_dim: int = 128, node_dim: int = 3, temporal_horizon: int = 0, legacy_depot_projection: bool = False)
:canonical: src.models.subnets.embeddings.mvptp.MVPTPInitEmbedding

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.embeddings.mvptp.MVPTPInitEmbedding
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.embeddings.mvptp.MVPTPInitEmbedding.__init__
```

````{py:method} forward(td: typing.Union[tensordict.TensorDict, typing.Dict[str, typing.Any]], temporal_features: bool = True) -> torch.Tensor
:canonical: src.models.subnets.embeddings.mvptp.MVPTPInitEmbedding.forward

```{autodoc2-docstring} src.models.subnets.embeddings.mvptp.MVPTPInitEmbedding.forward
```

````

````{py:method} init_node_embeddings(nodes: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.models.subnets.embeddings.mvptp.MVPTPInitEmbedding.init_node_embeddings

```{autodoc2-docstring} src.models.subnets.embeddings.mvptp.MVPTPInitEmbedding.init_node_embeddings
```

````

`````
