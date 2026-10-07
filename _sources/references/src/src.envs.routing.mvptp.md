# {py:mod}`src.envs.routing.mvptp`

```{py:module} src.envs.routing.mvptp
```

```{autodoc2-docstring} src.envs.routing.mvptp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MVPTPEnv <src.envs.routing.mvptp.MVPTPEnv>`
  - ```{autodoc2-docstring} src.envs.routing.mvptp.MVPTPEnv
    :summary:
    ```
````

### API

`````{py:class} MVPTPEnv(generator: typing.Optional[logic.src.envs.generators.PTPGenerator] = None, generator_params: typing.Optional[dict] = None, waste_weight: float = 1.0, cost_weight: float = 1.0, revenue_kg: typing.Optional[float] = None, cost_km: typing.Optional[float] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.routing.mvptp.MVPTPEnv

Bases: {py:obj}`logic.src.envs.routing.ptp.PTPEnv`

```{autodoc2-docstring} src.envs.routing.mvptp.MVPTPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.routing.mvptp.MVPTPEnv.__init__
```

````{py:attribute} name
:canonical: src.envs.routing.mvptp.MVPTPEnv.name
:type: str
:value: >
   'mvptp'

```{autodoc2-docstring} src.envs.routing.mvptp.MVPTPEnv.name
```

````

````{py:method} _reset_instance(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.mvptp.MVPTPEnv._reset_instance

```{autodoc2-docstring} src.envs.routing.mvptp.MVPTPEnv._reset_instance
```

````

````{py:method} _reset(tensordict: typing.Optional[tensordict.TensorDict] = None, **kwargs) -> tensordict.TensorDict
:canonical: src.envs.routing.mvptp.MVPTPEnv._reset

```{autodoc2-docstring} src.envs.routing.mvptp.MVPTPEnv._reset
```

````

````{py:method} _step_instance(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.mvptp.MVPTPEnv._step_instance

```{autodoc2-docstring} src.envs.routing.mvptp.MVPTPEnv._step_instance
```

````

````{py:method} _step(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.mvptp.MVPTPEnv._step

```{autodoc2-docstring} src.envs.routing.mvptp.MVPTPEnv._step
```

````

````{py:method} _get_action_mask(tensordict: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.mvptp.MVPTPEnv._get_action_mask

```{autodoc2-docstring} src.envs.routing.mvptp.MVPTPEnv._get_action_mask
```

````

`````
