# {py:mod}`src.envs.routing.cwcvrp`

```{py:module} src.envs.routing.cwcvrp
```

```{autodoc2-docstring} src.envs.routing.cwcvrp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CWCVRPEnv <src.envs.routing.cwcvrp.CWCVRPEnv>`
  - ```{autodoc2-docstring} src.envs.routing.cwcvrp.CWCVRPEnv
    :summary:
    ```
````

### API

`````{py:class} CWCVRPEnv(generator: typing.Optional[logic.src.envs.generators.WCVRPGenerator] = None, generator_params: typing.Optional[dict] = None, overflow_penalty: float = 10.0, waste_weight: float = 1.0, cost_weight: float = 1.0, revenue_kg: typing.Optional[float] = None, cost_km: typing.Optional[float] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.routing.cwcvrp.CWCVRPEnv

Bases: {py:obj}`logic.src.envs.routing.wcvrp.WCVRPEnv`

```{autodoc2-docstring} src.envs.routing.cwcvrp.CWCVRPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.routing.cwcvrp.CWCVRPEnv.__init__
```

````{py:attribute} name
:canonical: src.envs.routing.cwcvrp.CWCVRPEnv.name
:type: str
:value: >
   'cwcvrp'

```{autodoc2-docstring} src.envs.routing.cwcvrp.CWCVRPEnv.name
```

````

`````
