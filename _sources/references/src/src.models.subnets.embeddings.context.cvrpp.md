# {py:mod}`src.models.subnets.embeddings.context.cvrpp`

```{py:module} src.models.subnets.embeddings.context.cvrpp
```

```{autodoc2-docstring} src.models.subnets.embeddings.context.cvrpp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CVRPPContextEmbedder <src.models.subnets.embeddings.context.cvrpp.CVRPPContextEmbedder>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.context.cvrpp.CVRPPContextEmbedder
    :summary:
    ```
````

### API

`````{py:class} CVRPPContextEmbedder(embed_dim: int, node_dim: int = NODE_DIM, temporal_horizon: int = 0)
:canonical: src.models.subnets.embeddings.context.cvrpp.CVRPPContextEmbedder

Bases: {py:obj}`src.models.subnets.embeddings.context.base.ContextEmbedder`

```{autodoc2-docstring} src.models.subnets.embeddings.context.cvrpp.CVRPPContextEmbedder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.embeddings.context.cvrpp.CVRPPContextEmbedder.__init__
```

````{py:method} init_node_embeddings(nodes: typing.Dict[str, typing.Any], temporal_features: bool = True) -> torch.Tensor
:canonical: src.models.subnets.embeddings.context.cvrpp.CVRPPContextEmbedder.init_node_embeddings

```{autodoc2-docstring} src.models.subnets.embeddings.context.cvrpp.CVRPPContextEmbedder.init_node_embeddings
```

````

````{py:method} _step_context(embeddings: torch.Tensor, state: typing.Any) -> torch.Tensor
:canonical: src.models.subnets.embeddings.context.cvrpp.CVRPPContextEmbedder._step_context

```{autodoc2-docstring} src.models.subnets.embeddings.context.cvrpp.CVRPPContextEmbedder._step_context
```

````

````{py:property} step_context_dim
:canonical: src.models.subnets.embeddings.context.cvrpp.CVRPPContextEmbedder.step_context_dim
:type: int

```{autodoc2-docstring} src.models.subnets.embeddings.context.cvrpp.CVRPPContextEmbedder.step_context_dim
```

````

`````
