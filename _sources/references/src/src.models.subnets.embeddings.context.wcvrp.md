# {py:mod}`src.models.subnets.embeddings.context.wcvrp`

```{py:module} src.models.subnets.embeddings.context.wcvrp
```

```{autodoc2-docstring} src.models.subnets.embeddings.context.wcvrp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`WCVRPContextEmbedder <src.models.subnets.embeddings.context.wcvrp.WCVRPContextEmbedder>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.context.wcvrp.WCVRPContextEmbedder
    :summary:
    ```
````

### API

`````{py:class} WCVRPContextEmbedder(embed_dim: int, node_dim: int = NODE_DIM, temporal_horizon: int = 0)
:canonical: src.models.subnets.embeddings.context.wcvrp.WCVRPContextEmbedder

Bases: {py:obj}`src.models.subnets.embeddings.context.base.ContextEmbedder`

```{autodoc2-docstring} src.models.subnets.embeddings.context.wcvrp.WCVRPContextEmbedder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.embeddings.context.wcvrp.WCVRPContextEmbedder.__init__
```

````{py:method} init_node_embeddings(nodes: dict[str, typing.Any], temporal_features: bool = True) -> torch.Tensor
:canonical: src.models.subnets.embeddings.context.wcvrp.WCVRPContextEmbedder.init_node_embeddings

````

````{py:property} step_context_dim
:canonical: src.models.subnets.embeddings.context.wcvrp.WCVRPContextEmbedder.step_context_dim
:type: int

````

`````
