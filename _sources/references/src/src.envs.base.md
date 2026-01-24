# {py:mod}`src.envs.base`

```{py:module} src.envs.base
```

```{autodoc2-docstring} src.envs.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RL4COEnvBase <src.envs.base.RL4COEnvBase>`
  - ```{autodoc2-docstring} src.envs.base.RL4COEnvBase
    :summary:
    ```
* - {py:obj}`ImprovementEnvBase <src.envs.base.ImprovementEnvBase>`
  - ```{autodoc2-docstring} src.envs.base.ImprovementEnvBase
    :summary:
    ```
````

### API

`````{py:class} RL4COEnvBase(generator: typing.Optional[typing.Any] = None, generator_params: typing.Optional[dict] = None, device: typing.Union[str, torch.device] = 'cpu', batch_size: typing.Optional[typing.Union[torch.Size, int]] = None, **kwargs: typing.Any)
:canonical: src.envs.base.RL4COEnvBase

Bases: {py:obj}`torchrl.envs.EnvBase`

```{autodoc2-docstring} src.envs.base.RL4COEnvBase
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.base.RL4COEnvBase.__init__
```

````{py:attribute} name
:canonical: src.envs.base.RL4COEnvBase.name
:type: str
:value: >
   'base'

```{autodoc2-docstring} src.envs.base.RL4COEnvBase.name
```

````

````{py:method} _reset(td: typing.Optional[tensordict.TensorDict] = None, batch_size: typing.Optional[int] = None) -> tensordict.TensorDict
:canonical: src.envs.base.RL4COEnvBase._reset

```{autodoc2-docstring} src.envs.base.RL4COEnvBase._reset
```

````

````{py:method} _reset_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.base.RL4COEnvBase._reset_instance
:abstractmethod:

```{autodoc2-docstring} src.envs.base.RL4COEnvBase._reset_instance
```

````

````{py:method} _step(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.base.RL4COEnvBase._step

```{autodoc2-docstring} src.envs.base.RL4COEnvBase._step
```

````

````{py:method} _step_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.base.RL4COEnvBase._step_instance
:abstractmethod:

```{autodoc2-docstring} src.envs.base.RL4COEnvBase._step_instance
```

````

````{py:method} _get_reward(td: tensordict.TensorDict, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.base.RL4COEnvBase._get_reward
:abstractmethod:

```{autodoc2-docstring} src.envs.base.RL4COEnvBase._get_reward
```

````

````{py:method} _get_action_mask(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.base.RL4COEnvBase._get_action_mask
:abstractmethod:

```{autodoc2-docstring} src.envs.base.RL4COEnvBase._get_action_mask
```

````

````{py:method} _set_seed(seed: typing.Optional[int])
:canonical: src.envs.base.RL4COEnvBase._set_seed

```{autodoc2-docstring} src.envs.base.RL4COEnvBase._set_seed
```

````

````{py:method} _check_done(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.base.RL4COEnvBase._check_done

```{autodoc2-docstring} src.envs.base.RL4COEnvBase._check_done
```

````

````{py:method} get_reward(td: tensordict.TensorDict, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.base.RL4COEnvBase.get_reward

```{autodoc2-docstring} src.envs.base.RL4COEnvBase.get_reward
```

````

````{py:method} render(td: tensordict.TensorDict, **kwargs: typing.Any) -> typing.Any
:canonical: src.envs.base.RL4COEnvBase.render
:abstractmethod:

```{autodoc2-docstring} src.envs.base.RL4COEnvBase.render
```

````

````{py:method} __repr__() -> str
:canonical: src.envs.base.RL4COEnvBase.__repr__

```{autodoc2-docstring} src.envs.base.RL4COEnvBase.__repr__
```

````

`````

`````{py:class} ImprovementEnvBase(generator: typing.Optional[typing.Any] = None, generator_params: typing.Optional[dict] = None, device: typing.Union[str, torch.device] = 'cpu', batch_size: typing.Optional[typing.Union[torch.Size, int]] = None, **kwargs: typing.Any)
:canonical: src.envs.base.ImprovementEnvBase

Bases: {py:obj}`src.envs.base.RL4COEnvBase`

```{autodoc2-docstring} src.envs.base.ImprovementEnvBase
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.base.ImprovementEnvBase.__init__
```

````{py:attribute} name
:canonical: src.envs.base.ImprovementEnvBase.name
:type: str
:value: >
   'improvement_base'

```{autodoc2-docstring} src.envs.base.ImprovementEnvBase.name
```

````

````{py:method} _get_initial_solution(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.base.ImprovementEnvBase._get_initial_solution
:abstractmethod:

```{autodoc2-docstring} src.envs.base.ImprovementEnvBase._get_initial_solution
```

````

````{py:method} _reset_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.base.ImprovementEnvBase._reset_instance

```{autodoc2-docstring} src.envs.base.ImprovementEnvBase._reset_instance
```

````

`````
