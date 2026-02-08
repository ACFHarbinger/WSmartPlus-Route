# {py:mod}`src.models.subnets.embeddings.edges.base`

```{py:module} src.models.subnets.embeddings.edges.base
```

```{autodoc2-docstring} src.models.subnets.embeddings.edges.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EdgeEmbedding <src.models.subnets.embeddings.edges.base.EdgeEmbedding>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.edges.base.EdgeEmbedding
    :summary:
    ```
````

### API

`````{py:class} EdgeEmbedding(embed_dim: int, linear_bias: bool = True, sparsify: bool = True, k_sparse: int | collections.abc.Callable[[int], int] | None = None)
:canonical: src.models.subnets.embeddings.edges.base.EdgeEmbedding

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.embeddings.edges.base.EdgeEmbedding
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.embeddings.edges.base.EdgeEmbedding.__init__
```

````{py:attribute} node_dim
:canonical: src.models.subnets.embeddings.edges.base.EdgeEmbedding.node_dim
:type: int
:value: >
   1

```{autodoc2-docstring} src.models.subnets.embeddings.edges.base.EdgeEmbedding.node_dim
```

````

````{py:method} forward(td, init_embeddings: torch.Tensor)
:canonical: src.models.subnets.embeddings.edges.base.EdgeEmbedding.forward

```{autodoc2-docstring} src.models.subnets.embeddings.edges.base.EdgeEmbedding.forward
```

````

````{py:method} _cost_matrix_to_graph(batch_cost_matrix: torch.Tensor, init_embeddings: torch.Tensor)
:canonical: src.models.subnets.embeddings.edges.base.EdgeEmbedding._cost_matrix_to_graph

```{autodoc2-docstring} src.models.subnets.embeddings.edges.base.EdgeEmbedding._cost_matrix_to_graph
```

````

`````
