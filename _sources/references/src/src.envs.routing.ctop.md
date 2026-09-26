# {py:mod}`src.envs.routing.ctop`

```{py:module} src.envs.routing.ctop
```

```{autodoc2-docstring} src.envs.routing.ctop
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CTOPEnv <src.envs.routing.ctop.CTOPEnv>`
  - ```{autodoc2-docstring} src.envs.routing.ctop.CTOPEnv
    :summary:
    ```
````

### API

`````{py:class} CTOPEnv(generator: typing.Optional[logic.src.envs.generators.ctop.CTOPGenerator] = None, generator_params: typing.Optional[dict] = None, waste_weight: float = 1.0, cost_weight: float = 1.0, revenue_kg: typing.Optional[float] = None, cost_km: typing.Optional[float] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.routing.ctop.CTOPEnv

Bases: {py:obj}`logic.src.envs.routing.cvrpp.CVRPPEnv`

```{autodoc2-docstring} src.envs.routing.ctop.CTOPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.routing.ctop.CTOPEnv.__init__
```

````{py:attribute} name
:canonical: src.envs.routing.ctop.CTOPEnv.name
:type: str
:value: >
   'ctop'

```{autodoc2-docstring} src.envs.routing.ctop.CTOPEnv.name
```

````

````{py:method} _reset_instance(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.ctop.CTOPEnv._reset_instance

```{autodoc2-docstring} src.envs.routing.ctop.CTOPEnv._reset_instance
```

````

````{py:method} _reset(tensordict: typing.Optional[tensordict.TensorDict] = None, **kwargs) -> tensordict.TensorDict
:canonical: src.envs.routing.ctop.CTOPEnv._reset

```{autodoc2-docstring} src.envs.routing.ctop.CTOPEnv._reset
```

````

````{py:method} _step(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.ctop.CTOPEnv._step

```{autodoc2-docstring} src.envs.routing.ctop.CTOPEnv._step
```

````

````{py:method} _step_instance(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.ctop.CTOPEnv._step_instance

```{autodoc2-docstring} src.envs.routing.ctop.CTOPEnv._step_instance
```

````

````{py:method} _get_action_mask(tensordict: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.ctop.CTOPEnv._get_action_mask

```{autodoc2-docstring} src.envs.routing.ctop.CTOPEnv._get_action_mask
```

````

````{py:method} _check_done(tensordict: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.ctop.CTOPEnv._check_done

```{autodoc2-docstring} src.envs.routing.ctop.CTOPEnv._check_done
```

````

`````
