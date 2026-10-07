# {py:mod}`src.envs.routing.tcmvptp`

```{py:module} src.envs.routing.tcmvptp
```

```{autodoc2-docstring} src.envs.routing.tcmvptp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TCMVPTPEnv <src.envs.routing.tcmvptp.TCMVPTPEnv>`
  - ```{autodoc2-docstring} src.envs.routing.tcmvptp.TCMVPTPEnv
    :summary:
    ```
````

### API

`````{py:class} TCMVPTPEnv(generator: typing.Optional[logic.src.envs.generators.tcmvptp.TCMVPTPGenerator] = None, generator_params: typing.Optional[dict] = None, waste_weight: float = 1.0, cost_weight: float = 1.0, revenue_kg: typing.Optional[float] = None, cost_km: typing.Optional[float] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.routing.tcmvptp.TCMVPTPEnv

Bases: {py:obj}`logic.src.envs.routing.mvptp.MVPTPEnv`

```{autodoc2-docstring} src.envs.routing.tcmvptp.TCMVPTPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.routing.tcmvptp.TCMVPTPEnv.__init__
```

````{py:attribute} name
:canonical: src.envs.routing.tcmvptp.TCMVPTPEnv.name
:type: str
:value: >
   'tcmvptp'

```{autodoc2-docstring} src.envs.routing.tcmvptp.TCMVPTPEnv.name
```

````

````{py:method} _reset_instance(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.tcmvptp.TCMVPTPEnv._reset_instance

```{autodoc2-docstring} src.envs.routing.tcmvptp.TCMVPTPEnv._reset_instance
```

````

````{py:method} _reset(tensordict: typing.Optional[tensordict.TensorDict] = None, **kwargs) -> tensordict.TensorDict
:canonical: src.envs.routing.tcmvptp.TCMVPTPEnv._reset

```{autodoc2-docstring} src.envs.routing.tcmvptp.TCMVPTPEnv._reset
```

````

````{py:method} _step(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.tcmvptp.TCMVPTPEnv._step

```{autodoc2-docstring} src.envs.routing.tcmvptp.TCMVPTPEnv._step
```

````

````{py:method} _step_instance(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.tcmvptp.TCMVPTPEnv._step_instance

```{autodoc2-docstring} src.envs.routing.tcmvptp.TCMVPTPEnv._step_instance
```

````

````{py:method} _get_action_mask(tensordict: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.tcmvptp.TCMVPTPEnv._get_action_mask

```{autodoc2-docstring} src.envs.routing.tcmvptp.TCMVPTPEnv._get_action_mask
```

````

````{py:method} _check_done(tensordict: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.tcmvptp.TCMVPTPEnv._check_done

```{autodoc2-docstring} src.envs.routing.tcmvptp.TCMVPTPEnv._check_done
```

````

`````
