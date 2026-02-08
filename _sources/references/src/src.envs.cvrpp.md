# {py:mod}`src.envs.cvrpp`

```{py:module} src.envs.cvrpp
```

```{autodoc2-docstring} src.envs.cvrpp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CVRPPEnv <src.envs.cvrpp.CVRPPEnv>`
  - ```{autodoc2-docstring} src.envs.cvrpp.CVRPPEnv
    :summary:
    ```
````

### API

`````{py:class} CVRPPEnv(generator: typing.Optional[logic.src.envs.generators.VRPPGenerator] = None, generator_params: typing.Optional[dict] = None, waste_weight: float = 1.0, cost_weight: float = 1.0, revenue_kg: typing.Optional[float] = None, cost_km: typing.Optional[float] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.cvrpp.CVRPPEnv

Bases: {py:obj}`logic.src.envs.vrpp.VRPPEnv`

```{autodoc2-docstring} src.envs.cvrpp.CVRPPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.cvrpp.CVRPPEnv.__init__
```

````{py:attribute} name
:canonical: src.envs.cvrpp.CVRPPEnv.name
:type: str
:value: >
   'cvrpp'

```{autodoc2-docstring} src.envs.cvrpp.CVRPPEnv.name
```

````

````{py:method} _reset_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.cvrpp.CVRPPEnv._reset_instance

```{autodoc2-docstring} src.envs.cvrpp.CVRPPEnv._reset_instance
```

````

````{py:method} _reset(tensordict: typing.Optional[tensordict.TensorDict] = None, **kwargs) -> tensordict.TensorDict
:canonical: src.envs.cvrpp.CVRPPEnv._reset

````

````{py:method} _step_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.cvrpp.CVRPPEnv._step_instance

```{autodoc2-docstring} src.envs.cvrpp.CVRPPEnv._step_instance
```

````

````{py:method} _step(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.cvrpp.CVRPPEnv._step

````

````{py:method} _get_action_mask(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.cvrpp.CVRPPEnv._get_action_mask

```{autodoc2-docstring} src.envs.cvrpp.CVRPPEnv._get_action_mask
```

````

`````
