# {py:mod}`src.models.subnets.encoders.nargnn.edge_embedding`

```{py:module} src.models.subnets.encoders.nargnn.edge_embedding
```

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.edge_embedding
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SimplifiedEdgeEmbedding <src.models.subnets.encoders.nargnn.edge_embedding.SimplifiedEdgeEmbedding>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.nargnn.edge_embedding.SimplifiedEdgeEmbedding
    :summary:
    ```
````

### API

`````{py:class} SimplifiedEdgeEmbedding(embed_dim: int, k_sparse: typing.Optional[int] = None, linear_bias: bool = True)
:canonical: src.models.subnets.encoders.nargnn.edge_embedding.SimplifiedEdgeEmbedding

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.edge_embedding.SimplifiedEdgeEmbedding
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.edge_embedding.SimplifiedEdgeEmbedding.__init__
```

````{py:method} forward(td: tensordict.TensorDict, init_embeddings: torch.Tensor) -> torch_geometric.data.Batch
:canonical: src.models.subnets.encoders.nargnn.edge_embedding.SimplifiedEdgeEmbedding.forward

```{autodoc2-docstring} src.models.subnets.encoders.nargnn.edge_embedding.SimplifiedEdgeEmbedding.forward
```

````

`````
