# {py:mod}`src.models.policies.base`

```{py:module} src.models.policies.base
```

```{autodoc2-docstring} src.models.policies.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ConstructivePolicy <src.models.policies.base.ConstructivePolicy>`
  - ```{autodoc2-docstring} src.models.policies.base.ConstructivePolicy
    :summary:
    ```
* - {py:obj}`ImprovementPolicy <src.models.policies.base.ImprovementPolicy>`
  - ```{autodoc2-docstring} src.models.policies.base.ImprovementPolicy
    :summary:
    ```
````

### API

`````{py:class} ConstructivePolicy(encoder: typing.Optional[torch.nn.Module] = None, decoder: typing.Optional[torch.nn.Module] = None, env_name: typing.Optional[str] = None, embed_dim: int = 128, **kwargs)
:canonical: src.models.policies.base.ConstructivePolicy

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.policies.base.ConstructivePolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.base.ConstructivePolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'sampling', num_starts: int = 1, **kwargs) -> dict
:canonical: src.models.policies.base.ConstructivePolicy.forward
:abstractmethod:

```{autodoc2-docstring} src.models.policies.base.ConstructivePolicy.forward
```

````

````{py:method} _select_action(logits: torch.Tensor, mask: torch.Tensor, decode_type: str = 'sampling') -> tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.policies.base.ConstructivePolicy._select_action

```{autodoc2-docstring} src.models.policies.base.ConstructivePolicy._select_action
```

````

`````

`````{py:class} ImprovementPolicy(encoder: typing.Optional[torch.nn.Module] = None, decoder: typing.Optional[torch.nn.Module] = None, env_name: typing.Optional[str] = None, **kwargs)
:canonical: src.models.policies.base.ImprovementPolicy

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.policies.base.ImprovementPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.base.ImprovementPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, **kwargs) -> dict
:canonical: src.models.policies.base.ImprovementPolicy.forward
:abstractmethod:

```{autodoc2-docstring} src.models.policies.base.ImprovementPolicy.forward
```

````

`````
