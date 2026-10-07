# {py:mod}`src.models.subnets.embeddings.context.mvptp`

```{py:module} src.models.subnets.embeddings.context.mvptp
```

```{autodoc2-docstring} src.models.subnets.embeddings.context.mvptp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MVPTPContextEmbedder <src.models.subnets.embeddings.context.mvptp.MVPTPContextEmbedder>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.context.mvptp.MVPTPContextEmbedder
    :summary:
    ```
````

### API

`````{py:class} MVPTPContextEmbedder(embed_dim: int, node_dim: int = NODE_DIM, temporal_horizon: int = 0)
:canonical: src.models.subnets.embeddings.context.mvptp.MVPTPContextEmbedder

Bases: {py:obj}`src.models.subnets.embeddings.context.base.ContextEmbedder`

```{autodoc2-docstring} src.models.subnets.embeddings.context.mvptp.MVPTPContextEmbedder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.embeddings.context.mvptp.MVPTPContextEmbedder.__init__
```

````{py:method} init_node_embeddings(nodes: typing.Dict[str, typing.Any], temporal_features: bool = True) -> torch.Tensor
:canonical: src.models.subnets.embeddings.context.mvptp.MVPTPContextEmbedder.init_node_embeddings

```{autodoc2-docstring} src.models.subnets.embeddings.context.mvptp.MVPTPContextEmbedder.init_node_embeddings
```

````

````{py:method} _step_context(embeddings: torch.Tensor, state: typing.Any) -> torch.Tensor
:canonical: src.models.subnets.embeddings.context.mvptp.MVPTPContextEmbedder._step_context

```{autodoc2-docstring} src.models.subnets.embeddings.context.mvptp.MVPTPContextEmbedder._step_context
```

````

````{py:property} step_context_dim
:canonical: src.models.subnets.embeddings.context.mvptp.MVPTPContextEmbedder.step_context_dim
:type: int

```{autodoc2-docstring} src.models.subnets.embeddings.context.mvptp.MVPTPContextEmbedder.step_context_dim
```

````

`````
