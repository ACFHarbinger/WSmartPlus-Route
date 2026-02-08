# {py:mod}`src.models.subnets.embeddings.state.env`

```{py:module} src.models.subnets.embeddings.state.env
```

```{autodoc2-docstring} src.models.subnets.embeddings.state.env
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EnvState <src.models.subnets.embeddings.state.env.EnvState>`
  - ```{autodoc2-docstring} src.models.subnets.embeddings.state.env.EnvState
    :summary:
    ```
````

### API

`````{py:class} EnvState(embed_dim: int, step_context_dim: int = 0, node_dim: int = 0)
:canonical: src.models.subnets.embeddings.state.env.EnvState

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.embeddings.state.env.EnvState
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.embeddings.state.env.EnvState.__init__
```

````{py:method} forward(embeddings: torch.Tensor, td: typing.Any) -> torch.Tensor
:canonical: src.models.subnets.embeddings.state.env.EnvState.forward

```{autodoc2-docstring} src.models.subnets.embeddings.state.env.EnvState.forward
```

````

````{py:method} _cur_node_embedding(embeddings: torch.Tensor, td: typing.Any) -> torch.Tensor
:canonical: src.models.subnets.embeddings.state.env.EnvState._cur_node_embedding

```{autodoc2-docstring} src.models.subnets.embeddings.state.env.EnvState._cur_node_embedding
```

````

````{py:method} _state_embedding(embeddings: torch.Tensor, td: typing.Any) -> torch.Tensor
:canonical: src.models.subnets.embeddings.state.env.EnvState._state_embedding

```{autodoc2-docstring} src.models.subnets.embeddings.state.env.EnvState._state_embedding
```

````

`````
