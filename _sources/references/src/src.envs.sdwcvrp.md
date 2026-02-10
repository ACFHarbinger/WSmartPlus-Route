# {py:mod}`src.envs.sdwcvrp`

```{py:module} src.envs.sdwcvrp
```

```{autodoc2-docstring} src.envs.sdwcvrp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SDWCVRPEnv <src.envs.sdwcvrp.SDWCVRPEnv>`
  - ```{autodoc2-docstring} src.envs.sdwcvrp.SDWCVRPEnv
    :summary:
    ```
````

### API

`````{py:class} SDWCVRPEnv(generator: typing.Optional[logic.src.envs.generators.WCVRPGenerator] = None, generator_params: typing.Optional[dict] = None, overflow_penalty: float = 10.0, waste_weight: float = 1.0, cost_weight: float = 1.0, revenue_kg: typing.Optional[float] = None, cost_km: typing.Optional[float] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.sdwcvrp.SDWCVRPEnv

Bases: {py:obj}`logic.src.envs.wcvrp.WCVRPEnv`

```{autodoc2-docstring} src.envs.sdwcvrp.SDWCVRPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.sdwcvrp.SDWCVRPEnv.__init__
```

````{py:attribute} name
:canonical: src.envs.sdwcvrp.SDWCVRPEnv.name
:type: str
:value: >
   'sdwcvrp'

```{autodoc2-docstring} src.envs.sdwcvrp.SDWCVRPEnv.name
```

````

````{py:method} _step(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.sdwcvrp.SDWCVRPEnv._step

```{autodoc2-docstring} src.envs.sdwcvrp.SDWCVRPEnv._step
```

````

`````
