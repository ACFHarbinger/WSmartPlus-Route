# {py:mod}`src.models.subnets.embeddings.positional.absolute_positional_embedding`

```{py:module} src.models.subnets.embeddings.positional.absolute_positional_embedding
```

```{autodoc2-docstring} src.models.subnets.embeddings.positional.absolute_positional_embedding
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AbsolutePositionalEmbedding <src.models.subnets.embeddings.positional.absolute_positional_embedding.AbsolutePositionalEmbedding>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.positional.absolute_positional_embedding.AbsolutePositionalEmbedding
    :summary:
    ```
````

### API

`````{py:class} AbsolutePositionalEmbedding(embed_dim: int, max_len: int = 5000)
:canonical: src.models.subnets.embeddings.positional.absolute_positional_embedding.AbsolutePositionalEmbedding

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.embeddings.positional.absolute_positional_embedding.AbsolutePositionalEmbedding
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.embeddings.positional.absolute_positional_embedding.AbsolutePositionalEmbedding.__init__
```

````{py:attribute} pe
:canonical: src.models.subnets.embeddings.positional.absolute_positional_embedding.AbsolutePositionalEmbedding.pe
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} src.models.subnets.embeddings.positional.absolute_positional_embedding.AbsolutePositionalEmbedding.pe
```

````

````{py:method} forward(x: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.embeddings.positional.absolute_positional_embedding.AbsolutePositionalEmbedding.forward

```{autodoc2-docstring} src.models.subnets.embeddings.positional.absolute_positional_embedding.AbsolutePositionalEmbedding.forward
```

````

`````
