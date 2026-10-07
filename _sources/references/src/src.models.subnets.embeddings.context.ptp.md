# {py:mod}`src.models.subnets.embeddings.context.ptp`

```{py:module} src.models.subnets.embeddings.context.ptp
```

```{autodoc2-docstring} src.models.subnets.embeddings.context.ptp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PTPContextEmbedder <src.models.subnets.embeddings.context.ptp.PTPContextEmbedder>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.context.ptp.PTPContextEmbedder
    :summary:
    ```
````

### API

`````{py:class} PTPContextEmbedder(embed_dim: int, node_dim: int = NODE_DIM, temporal_horizon: int = 0)
:canonical: src.models.subnets.embeddings.context.ptp.PTPContextEmbedder

Bases: {py:obj}`src.models.subnets.embeddings.context.base.ContextEmbedder`

```{autodoc2-docstring} src.models.subnets.embeddings.context.ptp.PTPContextEmbedder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.embeddings.context.ptp.PTPContextEmbedder.__init__
```

````{py:method} init_node_embeddings(nodes: typing.Dict[str, typing.Any], temporal_features: bool = True) -> torch.Tensor
:canonical: src.models.subnets.embeddings.context.ptp.PTPContextEmbedder.init_node_embeddings

```{autodoc2-docstring} src.models.subnets.embeddings.context.ptp.PTPContextEmbedder.init_node_embeddings
```

````

````{py:method} _step_context(embeddings: torch.Tensor, state: typing.Any) -> torch.Tensor
:canonical: src.models.subnets.embeddings.context.ptp.PTPContextEmbedder._step_context

```{autodoc2-docstring} src.models.subnets.embeddings.context.ptp.PTPContextEmbedder._step_context
```

````

````{py:property} step_context_dim
:canonical: src.models.subnets.embeddings.context.ptp.PTPContextEmbedder.step_context_dim
:type: int

```{autodoc2-docstring} src.models.subnets.embeddings.context.ptp.PTPContextEmbedder.step_context_dim
```

````

`````
