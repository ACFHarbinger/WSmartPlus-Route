# {py:mod}`src.models.subnets.embeddings.dynamic`

```{py:module} src.models.subnets.embeddings.dynamic
```

```{autodoc2-docstring} src.models.subnets.embeddings.dynamic
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DynamicEmbedding <src.models.subnets.embeddings.dynamic.DynamicEmbedding>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.dynamic.DynamicEmbedding
    :summary:
    ```
````

### API

`````{py:class} DynamicEmbedding(embed_dim: int, dynamic_node_dim: int = 1)
:canonical: src.models.subnets.embeddings.dynamic.DynamicEmbedding

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.embeddings.dynamic.DynamicEmbedding
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.embeddings.dynamic.DynamicEmbedding.__init__
```

````{py:method} forward(td: typing.Any) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.embeddings.dynamic.DynamicEmbedding.forward

```{autodoc2-docstring} src.models.subnets.embeddings.dynamic.DynamicEmbedding.forward
```

````

`````
