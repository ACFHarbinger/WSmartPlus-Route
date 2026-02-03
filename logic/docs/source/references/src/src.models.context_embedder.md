# {py:mod}`src.models.context_embedder`

```{py:module} src.models.context_embedder
```

```{autodoc2-docstring} src.models.context_embedder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ContextEmbedder <src.models.context_embedder.ContextEmbedder>`
  - ```{autodoc2-docstring} src.models.context_embedder.ContextEmbedder
    :summary:
    ```
* - {py:obj}`WCContextEmbedder <src.models.context_embedder.WCContextEmbedder>`
  - ```{autodoc2-docstring} src.models.context_embedder.WCContextEmbedder
    :summary:
    ```
* - {py:obj}`VRPPContextEmbedder <src.models.context_embedder.VRPPContextEmbedder>`
  - ```{autodoc2-docstring} src.models.context_embedder.VRPPContextEmbedder
    :summary:
    ```
````

### API

`````{py:class} ContextEmbedder(embed_dim, node_dim, temporal_horizon)
:canonical: src.models.context_embedder.ContextEmbedder

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.context_embedder.ContextEmbedder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.context_embedder.ContextEmbedder.__init__
```

````{py:method} init_node_embeddings(input)
:canonical: src.models.context_embedder.ContextEmbedder.init_node_embeddings
:abstractmethod:

```{autodoc2-docstring} src.models.context_embedder.ContextEmbedder.init_node_embeddings
```

````

````{py:property} step_context_dim
:canonical: src.models.context_embedder.ContextEmbedder.step_context_dim
:abstractmethod:

```{autodoc2-docstring} src.models.context_embedder.ContextEmbedder.step_context_dim
```

````

`````

`````{py:class} WCContextEmbedder(embed_dim, node_dim=NODE_DIM, temporal_horizon=0)
:canonical: src.models.context_embedder.WCContextEmbedder

Bases: {py:obj}`src.models.context_embedder.ContextEmbedder`

```{autodoc2-docstring} src.models.context_embedder.WCContextEmbedder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.context_embedder.WCContextEmbedder.__init__
```

````{py:method} init_node_embeddings(nodes, temporal_features=True)
:canonical: src.models.context_embedder.WCContextEmbedder.init_node_embeddings

```{autodoc2-docstring} src.models.context_embedder.WCContextEmbedder.init_node_embeddings
```

````

````{py:property} step_context_dim
:canonical: src.models.context_embedder.WCContextEmbedder.step_context_dim

```{autodoc2-docstring} src.models.context_embedder.WCContextEmbedder.step_context_dim
```

````

`````

`````{py:class} VRPPContextEmbedder(embed_dim, node_dim=NODE_DIM, temporal_horizon=0)
:canonical: src.models.context_embedder.VRPPContextEmbedder

Bases: {py:obj}`src.models.context_embedder.ContextEmbedder`

```{autodoc2-docstring} src.models.context_embedder.VRPPContextEmbedder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.context_embedder.VRPPContextEmbedder.__init__
```

````{py:method} init_node_embeddings(nodes, temporal_features=True)
:canonical: src.models.context_embedder.VRPPContextEmbedder.init_node_embeddings

```{autodoc2-docstring} src.models.context_embedder.VRPPContextEmbedder.init_node_embeddings
```

````

````{py:property} step_context_dim
:canonical: src.models.context_embedder.VRPPContextEmbedder.step_context_dim

```{autodoc2-docstring} src.models.context_embedder.VRPPContextEmbedder.step_context_dim
```

````

`````
