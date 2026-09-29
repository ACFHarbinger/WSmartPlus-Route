# {py:mod}`src.models.subnets.embeddings.vrpp`

```{py:module} src.models.subnets.embeddings.vrpp
```

```{autodoc2-docstring} src.models.subnets.embeddings.vrpp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VRPPInitEmbedding <src.models.subnets.embeddings.vrpp.VRPPInitEmbedding>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.vrpp.VRPPInitEmbedding
    :summary:
    ```
````

### API

`````{py:class} VRPPInitEmbedding(embed_dim: int = 128, node_dim: int = 3, temporal_horizon: int = 0, legacy_depot_projection: bool = False)
:canonical: src.models.subnets.embeddings.vrpp.VRPPInitEmbedding

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.embeddings.vrpp.VRPPInitEmbedding
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.embeddings.vrpp.VRPPInitEmbedding.__init__
```

````{py:method} forward(td: typing.Union[tensordict.TensorDict, typing.Dict[str, typing.Any]], temporal_features: bool = True) -> torch.Tensor
:canonical: src.models.subnets.embeddings.vrpp.VRPPInitEmbedding.forward

```{autodoc2-docstring} src.models.subnets.embeddings.vrpp.VRPPInitEmbedding.forward
```

````

````{py:method} init_node_embeddings(nodes: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.models.subnets.embeddings.vrpp.VRPPInitEmbedding.init_node_embeddings

```{autodoc2-docstring} src.models.subnets.embeddings.vrpp.VRPPInitEmbedding.init_node_embeddings
```

````

`````
