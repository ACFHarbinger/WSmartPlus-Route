# {py:mod}`src.envs.routing.cvrpp`

```{py:module} src.envs.routing.cvrpp
```

```{autodoc2-docstring} src.envs.routing.cvrpp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CVRPPEnv <src.envs.routing.cvrpp.CVRPPEnv>`
  - ```{autodoc2-docstring} src.envs.routing.cvrpp.CVRPPEnv
    :summary:
    ```
````

### API

`````{py:class} CVRPPEnv(generator: typing.Optional[logic.src.envs.generators.VRPPGenerator] = None, generator_params: typing.Optional[dict] = None, waste_weight: float = 1.0, cost_weight: float = 1.0, revenue_kg: typing.Optional[float] = None, cost_km: typing.Optional[float] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.routing.cvrpp.CVRPPEnv

Bases: {py:obj}`logic.src.envs.routing.vrpp.VRPPEnv`

```{autodoc2-docstring} src.envs.routing.cvrpp.CVRPPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.routing.cvrpp.CVRPPEnv.__init__
```

````{py:attribute} name
:canonical: src.envs.routing.cvrpp.CVRPPEnv.name
:type: str
:value: >
   'cvrpp'

```{autodoc2-docstring} src.envs.routing.cvrpp.CVRPPEnv.name
```

````

````{py:method} _reset_instance(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.cvrpp.CVRPPEnv._reset_instance

```{autodoc2-docstring} src.envs.routing.cvrpp.CVRPPEnv._reset_instance
```

````

````{py:method} _reset(tensordict: typing.Optional[tensordict.TensorDict] = None, **kwargs) -> tensordict.TensorDict
:canonical: src.envs.routing.cvrpp.CVRPPEnv._reset

```{autodoc2-docstring} src.envs.routing.cvrpp.CVRPPEnv._reset
```

````

````{py:method} _step_instance(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.cvrpp.CVRPPEnv._step_instance

```{autodoc2-docstring} src.envs.routing.cvrpp.CVRPPEnv._step_instance
```

````

````{py:method} _step(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.cvrpp.CVRPPEnv._step

```{autodoc2-docstring} src.envs.routing.cvrpp.CVRPPEnv._step
```

````

````{py:method} _get_action_mask(tensordict: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.cvrpp.CVRPPEnv._get_action_mask

```{autodoc2-docstring} src.envs.routing.cvrpp.CVRPPEnv._get_action_mask
```

````

`````
