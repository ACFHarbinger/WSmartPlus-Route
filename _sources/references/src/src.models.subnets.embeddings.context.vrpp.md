# {py:mod}`src.models.subnets.embeddings.context.vrpp`

```{py:module} src.models.subnets.embeddings.context.vrpp
```

```{autodoc2-docstring} src.models.subnets.embeddings.context.vrpp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VRPPContextEmbedder <src.models.subnets.embeddings.context.vrpp.VRPPContextEmbedder>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.context.vrpp.VRPPContextEmbedder
    :summary:
    ```
````

### API

`````{py:class} VRPPContextEmbedder(embed_dim: int, node_dim: int = NODE_DIM, temporal_horizon: int = 0)
:canonical: src.models.subnets.embeddings.context.vrpp.VRPPContextEmbedder

Bases: {py:obj}`src.models.subnets.embeddings.context.base.ContextEmbedder`

```{autodoc2-docstring} src.models.subnets.embeddings.context.vrpp.VRPPContextEmbedder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.embeddings.context.vrpp.VRPPContextEmbedder.__init__
```

````{py:method} init_node_embeddings(nodes: dict[str, typing.Any], temporal_features: bool = True) -> torch.Tensor
:canonical: src.models.subnets.embeddings.context.vrpp.VRPPContextEmbedder.init_node_embeddings

````

````{py:method} _step_context(embeddings: torch.Tensor, state: typing.Any) -> torch.Tensor
:canonical: src.models.subnets.embeddings.context.vrpp.VRPPContextEmbedder._step_context

```{autodoc2-docstring} src.models.subnets.embeddings.context.vrpp.VRPPContextEmbedder._step_context
```

````

````{py:property} step_context_dim
:canonical: src.models.subnets.embeddings.context.vrpp.VRPPContextEmbedder.step_context_dim
:type: int

```{autodoc2-docstring} src.models.subnets.embeddings.context.vrpp.VRPPContextEmbedder.step_context_dim
```

````

`````
