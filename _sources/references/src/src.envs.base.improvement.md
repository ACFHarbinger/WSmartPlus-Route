# {py:mod}`src.envs.base.improvement`

```{py:module} src.envs.base.improvement
```

```{autodoc2-docstring} src.envs.base.improvement
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImprovementEnvBase <src.envs.base.improvement.ImprovementEnvBase>`
  - ```{autodoc2-docstring} src.envs.base.improvement.ImprovementEnvBase
    :summary:
    ```
````

### API

`````{py:class} ImprovementEnvBase(generator: typing.Optional[typing.Any] = None, generator_params: typing.Optional[dict] = None, device: typing.Union[str, torch.device] = 'cpu', batch_size: typing.Optional[typing.Union[torch.Size, int]] = None, **kwargs: typing.Any)
:canonical: src.envs.base.improvement.ImprovementEnvBase

Bases: {py:obj}`src.envs.base.base.RL4COEnvBase`

```{autodoc2-docstring} src.envs.base.improvement.ImprovementEnvBase
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.base.improvement.ImprovementEnvBase.__init__
```

````{py:attribute} name
:canonical: src.envs.base.improvement.ImprovementEnvBase.name
:type: str
:value: >
   'improvement_base'

```{autodoc2-docstring} src.envs.base.improvement.ImprovementEnvBase.name
```

````

````{py:method} _get_initial_solution(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.base.improvement.ImprovementEnvBase._get_initial_solution
:abstractmethod:

```{autodoc2-docstring} src.envs.base.improvement.ImprovementEnvBase._get_initial_solution
```

````

````{py:method} _reset_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.base.improvement.ImprovementEnvBase._reset_instance

```{autodoc2-docstring} src.envs.base.improvement.ImprovementEnvBase._reset_instance
```

````

`````
