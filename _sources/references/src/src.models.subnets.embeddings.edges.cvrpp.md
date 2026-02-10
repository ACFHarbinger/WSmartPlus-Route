# {py:mod}`src.models.subnets.embeddings.edges.cvrpp`

```{py:module} src.models.subnets.embeddings.edges.cvrpp
```

```{autodoc2-docstring} src.models.subnets.embeddings.edges.cvrpp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CVRPPEdgeEmbedding <src.models.subnets.embeddings.edges.cvrpp.CVRPPEdgeEmbedding>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.edges.cvrpp.CVRPPEdgeEmbedding
    :summary:
    ```
````

### API

`````{py:class} CVRPPEdgeEmbedding(embed_dim: int, linear_bias: bool = True, sparsify: bool = True, k_sparse: int | collections.abc.Callable[[int], int] | None = None)
:canonical: src.models.subnets.embeddings.edges.cvrpp.CVRPPEdgeEmbedding

Bases: {py:obj}`src.models.subnets.embeddings.edges.base.EdgeEmbedding`

```{autodoc2-docstring} src.models.subnets.embeddings.edges.cvrpp.CVRPPEdgeEmbedding
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.embeddings.edges.cvrpp.CVRPPEdgeEmbedding.__init__
```

````{py:method} _cost_matrix_to_graph(batch_cost_matrix: torch.Tensor, init_embeddings: torch.Tensor)
:canonical: src.models.subnets.embeddings.edges.cvrpp.CVRPPEdgeEmbedding._cost_matrix_to_graph

```{autodoc2-docstring} src.models.subnets.embeddings.edges.cvrpp.CVRPPEdgeEmbedding._cost_matrix_to_graph
```

````

`````
