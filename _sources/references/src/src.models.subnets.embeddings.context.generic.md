# {py:mod}`src.models.subnets.embeddings.context.generic`

```{py:module} src.models.subnets.embeddings.context.generic
```

```{autodoc2-docstring} src.models.subnets.embeddings.context.generic
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GenericContextEmbedder <src.models.subnets.embeddings.context.generic.GenericContextEmbedder>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.context.generic.GenericContextEmbedder
    :summary:
    ```
````

### API

`````{py:class} GenericContextEmbedder(embed_dim: int, node_dim: int = NODE_DIM, temporal_horizon: int = 0)
:canonical: src.models.subnets.embeddings.context.generic.GenericContextEmbedder

Bases: {py:obj}`src.models.subnets.embeddings.context.base.ContextEmbedder`

```{autodoc2-docstring} src.models.subnets.embeddings.context.generic.GenericContextEmbedder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.embeddings.context.generic.GenericContextEmbedder.__init__
```

````{py:method} init_node_embeddings(nodes: dict[str, typing.Any]) -> torch.Tensor
:canonical: src.models.subnets.embeddings.context.generic.GenericContextEmbedder.init_node_embeddings

```{autodoc2-docstring} src.models.subnets.embeddings.context.generic.GenericContextEmbedder.init_node_embeddings
```

````

````{py:method} _step_context(embeddings: torch.Tensor, state: typing.Any) -> torch.Tensor
:canonical: src.models.subnets.embeddings.context.generic.GenericContextEmbedder._step_context

```{autodoc2-docstring} src.models.subnets.embeddings.context.generic.GenericContextEmbedder._step_context
```

````

````{py:property} step_context_dim
:canonical: src.models.subnets.embeddings.context.generic.GenericContextEmbedder.step_context_dim
:type: int

```{autodoc2-docstring} src.models.subnets.embeddings.context.generic.GenericContextEmbedder.step_context_dim
```

````

`````
