# {py:mod}`src.envs.routing.ptp`

```{py:module} src.envs.routing.ptp
```

```{autodoc2-docstring} src.envs.routing.ptp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PTPEnv <src.envs.routing.ptp.PTPEnv>`
  - ```{autodoc2-docstring} src.envs.routing.ptp.PTPEnv
    :summary:
    ```
````

### API

`````{py:class} PTPEnv(generator: typing.Optional[logic.src.envs.generators.PTPGenerator] = None, generator_params: typing.Optional[dict] = None, waste_weight: float = 1.0, cost_weight: float = 1.0, revenue_kg: typing.Optional[float] = None, cost_km: typing.Optional[float] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.routing.ptp.PTPEnv

Bases: {py:obj}`logic.src.envs.base.base.RL4COEnvBase`

```{autodoc2-docstring} src.envs.routing.ptp.PTPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.routing.ptp.PTPEnv.__init__
```

````{py:attribute} NAME
:canonical: src.envs.routing.ptp.PTPEnv.NAME
:value: >
   'ptp'

```{autodoc2-docstring} src.envs.routing.ptp.PTPEnv.NAME
```

````

````{py:attribute} name
:canonical: src.envs.routing.ptp.PTPEnv.name
:type: str
:value: >
   'ptp'

```{autodoc2-docstring} src.envs.routing.ptp.PTPEnv.name
```

````

````{py:method} _reset_instance(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.ptp.PTPEnv._reset_instance

```{autodoc2-docstring} src.envs.routing.ptp.PTPEnv._reset_instance
```

````

````{py:method} _step(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.ptp.PTPEnv._step

```{autodoc2-docstring} src.envs.routing.ptp.PTPEnv._step
```

````

````{py:method} _step_instance(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.ptp.PTPEnv._step_instance

```{autodoc2-docstring} src.envs.routing.ptp.PTPEnv._step_instance
```

````

````{py:method} _get_action_mask(tensordict: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.ptp.PTPEnv._get_action_mask

```{autodoc2-docstring} src.envs.routing.ptp.PTPEnv._get_action_mask
```

````

````{py:method} _get_reward(tensordict: tensordict.TensorDictBase, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.routing.ptp.PTPEnv._get_reward

```{autodoc2-docstring} src.envs.routing.ptp.PTPEnv._get_reward
```

````

````{py:method} _check_done(tensordict: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.ptp.PTPEnv._check_done

```{autodoc2-docstring} src.envs.routing.ptp.PTPEnv._check_done
```

````

`````
