# {py:mod}`src.models.common.autoregressive.constructive`

```{py:module} src.models.common.autoregressive.constructive
```

```{autodoc2-docstring} src.models.common.autoregressive.constructive
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ConstructivePolicy <src.models.common.autoregressive.constructive.ConstructivePolicy>`
  - ```{autodoc2-docstring} src.models.common.autoregressive.constructive.ConstructivePolicy
    :summary:
    ```
````

### API

`````{py:class} ConstructivePolicy(encoder: typing.Optional[torch.nn.Module] = None, decoder: typing.Optional[torch.nn.Module] = None, env_name: typing.Optional[str] = None, embed_dim: int = 128, seed: int = 42, device: str = 'cpu', **kwargs)
:canonical: src.models.common.autoregressive.constructive.ConstructivePolicy

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.common.autoregressive.constructive.ConstructivePolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.autoregressive.constructive.ConstructivePolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, strategy: str = 'sampling', num_starts: int = 1, **kwargs) -> dict
:canonical: src.models.common.autoregressive.constructive.ConstructivePolicy.forward
:abstractmethod:

```{autodoc2-docstring} src.models.common.autoregressive.constructive.ConstructivePolicy.forward
```

````

````{py:method} _select_action(logits: torch.Tensor, mask: torch.Tensor, strategy: str = 'sampling', **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.models.common.autoregressive.constructive.ConstructivePolicy._select_action

```{autodoc2-docstring} src.models.common.autoregressive.constructive.ConstructivePolicy._select_action
```

````

`````
