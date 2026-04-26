# {py:mod}`src.envs.routing.wcvrp`

```{py:module} src.envs.routing.wcvrp
```

```{autodoc2-docstring} src.envs.routing.wcvrp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`WCVRPEnv <src.envs.routing.wcvrp.WCVRPEnv>`
  - ```{autodoc2-docstring} src.envs.routing.wcvrp.WCVRPEnv
    :summary:
    ```
````

### API

`````{py:class} WCVRPEnv(generator: typing.Optional[logic.src.envs.generators.WCVRPGenerator] = None, generator_params: typing.Optional[dict] = None, overflow_penalty: float = 10.0, waste_weight: float = 1.0, cost_weight: float = 1.0, revenue_kg: typing.Optional[float] = None, cost_km: typing.Optional[float] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.routing.wcvrp.WCVRPEnv

Bases: {py:obj}`logic.src.envs.base.base.RL4COEnvBase`

```{autodoc2-docstring} src.envs.routing.wcvrp.WCVRPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.routing.wcvrp.WCVRPEnv.__init__
```

````{py:attribute} NAME
:canonical: src.envs.routing.wcvrp.WCVRPEnv.NAME
:value: >
   'wcvrp'

```{autodoc2-docstring} src.envs.routing.wcvrp.WCVRPEnv.NAME
```

````

````{py:attribute} name
:canonical: src.envs.routing.wcvrp.WCVRPEnv.name
:type: str
:value: >
   'wcvrp'

```{autodoc2-docstring} src.envs.routing.wcvrp.WCVRPEnv.name
```

````

````{py:method} _reset_instance(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.wcvrp.WCVRPEnv._reset_instance

```{autodoc2-docstring} src.envs.routing.wcvrp.WCVRPEnv._reset_instance
```

````

````{py:method} _step_instance(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.wcvrp.WCVRPEnv._step_instance

```{autodoc2-docstring} src.envs.routing.wcvrp.WCVRPEnv._step_instance
```

````

````{py:method} _step(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.wcvrp.WCVRPEnv._step

```{autodoc2-docstring} src.envs.routing.wcvrp.WCVRPEnv._step
```

````

````{py:method} _get_action_mask(tensordict: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.wcvrp.WCVRPEnv._get_action_mask

```{autodoc2-docstring} src.envs.routing.wcvrp.WCVRPEnv._get_action_mask
```

````

````{py:method} _get_reward(tensordict: tensordict.TensorDictBase, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.routing.wcvrp.WCVRPEnv._get_reward

```{autodoc2-docstring} src.envs.routing.wcvrp.WCVRPEnv._get_reward
```

````

`````
