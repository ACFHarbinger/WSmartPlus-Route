# {py:mod}`src.models.subnets.embeddings.context.base`

```{py:module} src.models.subnets.embeddings.context.base
```

```{autodoc2-docstring} src.models.subnets.embeddings.context.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ContextEmbedder <src.models.subnets.embeddings.context.base.ContextEmbedder>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.context.base.ContextEmbedder
    :summary:
    ```
````

### API

`````{py:class} ContextEmbedder(embed_dim: int, node_dim: int, temporal_horizon: int)
:canonical: src.models.subnets.embeddings.context.base.ContextEmbedder

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.subnets.embeddings.context.base.ContextEmbedder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.embeddings.context.base.ContextEmbedder.__init__
```

````{py:method} init_node_embeddings(nodes: dict[str, typing.Any]) -> torch.Tensor
:canonical: src.models.subnets.embeddings.context.base.ContextEmbedder.init_node_embeddings
:abstractmethod:

```{autodoc2-docstring} src.models.subnets.embeddings.context.base.ContextEmbedder.init_node_embeddings
```

````

````{py:property} step_context_dim
:canonical: src.models.subnets.embeddings.context.base.ContextEmbedder.step_context_dim
:abstractmethod:
:type: int

```{autodoc2-docstring} src.models.subnets.embeddings.context.base.ContextEmbedder.step_context_dim
```

````

````{py:method} forward(input: dict[str, typing.Any]) -> torch.Tensor
:canonical: src.models.subnets.embeddings.context.base.ContextEmbedder.forward

```{autodoc2-docstring} src.models.subnets.embeddings.context.base.ContextEmbedder.forward
```

````

`````
