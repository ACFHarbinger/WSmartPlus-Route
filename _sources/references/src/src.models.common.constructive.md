# {py:mod}`src.models.common.constructive`

```{py:module} src.models.common.constructive
```

```{autodoc2-docstring} src.models.common.constructive
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ConstructivePolicy <src.models.common.constructive.ConstructivePolicy>`
  - ```{autodoc2-docstring} src.models.common.constructive.ConstructivePolicy
    :summary:
    ```
````

### API

`````{py:class} ConstructivePolicy(encoder: typing.Optional[torch.nn.Module] = None, decoder: typing.Optional[torch.nn.Module] = None, env_name: typing.Optional[str] = None, embed_dim: int = 128, **kwargs)
:canonical: src.models.common.constructive.ConstructivePolicy

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.common.constructive.ConstructivePolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.constructive.ConstructivePolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'sampling', num_starts: int = 1, **kwargs) -> dict
:canonical: src.models.common.constructive.ConstructivePolicy.forward
:abstractmethod:

```{autodoc2-docstring} src.models.common.constructive.ConstructivePolicy.forward
```

````

````{py:method} _select_action(logits: torch.Tensor, mask: torch.Tensor, decode_type: str = 'sampling', **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.models.common.constructive.ConstructivePolicy._select_action

```{autodoc2-docstring} src.models.common.constructive.ConstructivePolicy._select_action
```

````

`````
