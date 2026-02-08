# {py:mod}`src.models.subnets.embeddings.positional.cyclic_positional_embedding`

```{py:module} src.models.subnets.embeddings.positional.cyclic_positional_embedding
```

```{autodoc2-docstring} src.models.subnets.embeddings.positional.cyclic_positional_embedding
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CyclicPositionalEmbedding <src.models.subnets.embeddings.positional.cyclic_positional_embedding.CyclicPositionalEmbedding>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.positional.cyclic_positional_embedding.CyclicPositionalEmbedding
    :summary:
    ```
````

### API

`````{py:class} CyclicPositionalEmbedding(embed_dim: int, mean_pooling: bool = True)
:canonical: src.models.subnets.embeddings.positional.cyclic_positional_embedding.CyclicPositionalEmbedding

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.embeddings.positional.cyclic_positional_embedding.CyclicPositionalEmbedding
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.embeddings.positional.cyclic_positional_embedding.CyclicPositionalEmbedding.__init__
```

````{py:method} forward(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.embeddings.positional.cyclic_positional_embedding.CyclicPositionalEmbedding.forward

```{autodoc2-docstring} src.models.subnets.embeddings.positional.cyclic_positional_embedding.CyclicPositionalEmbedding.forward
```

````

`````
