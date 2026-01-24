# {py:mod}`src.envs.wcvrp`

```{py:module} src.envs.wcvrp
```

```{autodoc2-docstring} src.envs.wcvrp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`WCVRPEnv <src.envs.wcvrp.WCVRPEnv>`
  - ```{autodoc2-docstring} src.envs.wcvrp.WCVRPEnv
    :summary:
    ```
* - {py:obj}`CWCVRPEnv <src.envs.wcvrp.CWCVRPEnv>`
  - ```{autodoc2-docstring} src.envs.wcvrp.CWCVRPEnv
    :summary:
    ```
* - {py:obj}`SDWCVRPEnv <src.envs.wcvrp.SDWCVRPEnv>`
  - ```{autodoc2-docstring} src.envs.wcvrp.SDWCVRPEnv
    :summary:
    ```
````

### API

`````{py:class} WCVRPEnv(generator: typing.Optional[logic.src.envs.generators.WCVRPGenerator] = None, generator_params: typing.Optional[dict] = None, overflow_penalty: float = 10.0, collection_reward: float = 1.0, cost_weight: float = 1.0, revenue_kg: typing.Optional[float] = None, cost_km: typing.Optional[float] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.wcvrp.WCVRPEnv

Bases: {py:obj}`logic.src.envs.base.RL4COEnvBase`

```{autodoc2-docstring} src.envs.wcvrp.WCVRPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.wcvrp.WCVRPEnv.__init__
```

````{py:attribute} name
:canonical: src.envs.wcvrp.WCVRPEnv.name
:type: str
:value: >
   'wcvrp'

```{autodoc2-docstring} src.envs.wcvrp.WCVRPEnv.name
```

````

````{py:method} _reset_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.wcvrp.WCVRPEnv._reset_instance

```{autodoc2-docstring} src.envs.wcvrp.WCVRPEnv._reset_instance
```

````

````{py:method} _step_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.wcvrp.WCVRPEnv._step_instance

```{autodoc2-docstring} src.envs.wcvrp.WCVRPEnv._step_instance
```

````

````{py:method} _step(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.wcvrp.WCVRPEnv._step

````

````{py:method} _get_action_mask(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.wcvrp.WCVRPEnv._get_action_mask

```{autodoc2-docstring} src.envs.wcvrp.WCVRPEnv._get_action_mask
```

````

````{py:method} _get_reward(td: tensordict.TensorDict, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.wcvrp.WCVRPEnv._get_reward

```{autodoc2-docstring} src.envs.wcvrp.WCVRPEnv._get_reward
```

````

`````

`````{py:class} CWCVRPEnv(generator: typing.Optional[logic.src.envs.generators.WCVRPGenerator] = None, generator_params: typing.Optional[dict] = None, overflow_penalty: float = 10.0, collection_reward: float = 1.0, cost_weight: float = 1.0, revenue_kg: typing.Optional[float] = None, cost_km: typing.Optional[float] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.wcvrp.CWCVRPEnv

Bases: {py:obj}`src.envs.wcvrp.WCVRPEnv`

```{autodoc2-docstring} src.envs.wcvrp.CWCVRPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.wcvrp.CWCVRPEnv.__init__
```

````{py:attribute} name
:canonical: src.envs.wcvrp.CWCVRPEnv.name
:type: str
:value: >
   'cwcvrp'

```{autodoc2-docstring} src.envs.wcvrp.CWCVRPEnv.name
```

````

`````

`````{py:class} SDWCVRPEnv(generator: typing.Optional[logic.src.envs.generators.WCVRPGenerator] = None, generator_params: typing.Optional[dict] = None, overflow_penalty: float = 10.0, collection_reward: float = 1.0, cost_weight: float = 1.0, revenue_kg: typing.Optional[float] = None, cost_km: typing.Optional[float] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.wcvrp.SDWCVRPEnv

Bases: {py:obj}`src.envs.wcvrp.WCVRPEnv`

```{autodoc2-docstring} src.envs.wcvrp.SDWCVRPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.wcvrp.SDWCVRPEnv.__init__
```

````{py:attribute} name
:canonical: src.envs.wcvrp.SDWCVRPEnv.name
:type: str
:value: >
   'sdwcvrp'

```{autodoc2-docstring} src.envs.wcvrp.SDWCVRPEnv.name
```

````

````{py:method} _step(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.wcvrp.SDWCVRPEnv._step

```{autodoc2-docstring} src.envs.wcvrp.SDWCVRPEnv._step
```

````

`````
