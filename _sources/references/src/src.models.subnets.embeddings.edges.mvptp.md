# {py:mod}`src.models.subnets.embeddings.edges.mvptp`

```{py:module} src.models.subnets.embeddings.edges.mvptp
```

```{autodoc2-docstring} src.models.subnets.embeddings.edges.mvptp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MVPTPEdgeEmbedding <src.models.subnets.embeddings.edges.mvptp.MVPTPEdgeEmbedding>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.edges.mvptp.MVPTPEdgeEmbedding
    :summary:
    ```
````

### API

`````{py:class} MVPTPEdgeEmbedding(embed_dim: int, linear_bias: bool = True, sparsify: bool = True, k_sparse: typing.Optional[typing.Union[int, collections.abc.Callable[[int], int]]] = None)
:canonical: src.models.subnets.embeddings.edges.mvptp.MVPTPEdgeEmbedding

Bases: {py:obj}`src.models.subnets.embeddings.edges.base.EdgeEmbedding`

```{autodoc2-docstring} src.models.subnets.embeddings.edges.mvptp.MVPTPEdgeEmbedding
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.embeddings.edges.mvptp.MVPTPEdgeEmbedding.__init__
```

````{py:method} _cost_matrix_to_graph(batch_cost_matrix: torch.Tensor, init_embeddings: torch.Tensor) -> torch_geometric.data.Batch
:canonical: src.models.subnets.embeddings.edges.mvptp.MVPTPEdgeEmbedding._cost_matrix_to_graph

```{autodoc2-docstring} src.models.subnets.embeddings.edges.mvptp.MVPTPEdgeEmbedding._cost_matrix_to_graph
```

````

`````
