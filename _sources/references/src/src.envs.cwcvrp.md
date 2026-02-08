# {py:mod}`src.envs.cwcvrp`

```{py:module} src.envs.cwcvrp
```

```{autodoc2-docstring} src.envs.cwcvrp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CWCVRPEnv <src.envs.cwcvrp.CWCVRPEnv>`
  - ```{autodoc2-docstring} src.envs.cwcvrp.CWCVRPEnv
    :summary:
    ```
````

### API

`````{py:class} CWCVRPEnv(generator: typing.Optional[logic.src.envs.generators.WCVRPGenerator] = None, generator_params: typing.Optional[dict] = None, overflow_penalty: float = 10.0, collection_reward: float = 1.0, cost_weight: float = 1.0, revenue_kg: typing.Optional[float] = None, cost_km: typing.Optional[float] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.cwcvrp.CWCVRPEnv

Bases: {py:obj}`logic.src.envs.wcvrp.WCVRPEnv`

```{autodoc2-docstring} src.envs.cwcvrp.CWCVRPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.cwcvrp.CWCVRPEnv.__init__
```

````{py:attribute} name
:canonical: src.envs.cwcvrp.CWCVRPEnv.name
:type: str
:value: >
   'cwcvrp'

```{autodoc2-docstring} src.envs.cwcvrp.CWCVRPEnv.name
```

````

`````
