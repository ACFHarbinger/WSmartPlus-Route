# {py:mod}`src.envs.routing.vrpp`

```{py:module} src.envs.routing.vrpp
```

```{autodoc2-docstring} src.envs.routing.vrpp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VRPPEnv <src.envs.routing.vrpp.VRPPEnv>`
  - ```{autodoc2-docstring} src.envs.routing.vrpp.VRPPEnv
    :summary:
    ```
````

### API

`````{py:class} VRPPEnv(generator: typing.Optional[logic.src.envs.generators.VRPPGenerator] = None, generator_params: typing.Optional[dict] = None, waste_weight: float = 1.0, cost_weight: float = 1.0, revenue_kg: typing.Optional[float] = None, cost_km: typing.Optional[float] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.routing.vrpp.VRPPEnv

Bases: {py:obj}`logic.src.envs.base.base.RL4COEnvBase`

```{autodoc2-docstring} src.envs.routing.vrpp.VRPPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.routing.vrpp.VRPPEnv.__init__
```

````{py:attribute} NAME
:canonical: src.envs.routing.vrpp.VRPPEnv.NAME
:value: >
   'vrpp'

```{autodoc2-docstring} src.envs.routing.vrpp.VRPPEnv.NAME
```

````

````{py:attribute} name
:canonical: src.envs.routing.vrpp.VRPPEnv.name
:type: str
:value: >
   'vrpp'

```{autodoc2-docstring} src.envs.routing.vrpp.VRPPEnv.name
```

````

````{py:method} _reset_instance(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.vrpp.VRPPEnv._reset_instance

```{autodoc2-docstring} src.envs.routing.vrpp.VRPPEnv._reset_instance
```

````

````{py:method} _step(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.vrpp.VRPPEnv._step

```{autodoc2-docstring} src.envs.routing.vrpp.VRPPEnv._step
```

````

````{py:method} _step_instance(tensordict: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.vrpp.VRPPEnv._step_instance

```{autodoc2-docstring} src.envs.routing.vrpp.VRPPEnv._step_instance
```

````

````{py:method} _get_action_mask(tensordict: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.vrpp.VRPPEnv._get_action_mask

```{autodoc2-docstring} src.envs.routing.vrpp.VRPPEnv._get_action_mask
```

````

````{py:method} _get_reward(tensordict: tensordict.TensorDictBase, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.routing.vrpp.VRPPEnv._get_reward

```{autodoc2-docstring} src.envs.routing.vrpp.VRPPEnv._get_reward
```

````

````{py:method} _check_done(tensordict: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.vrpp.VRPPEnv._check_done

```{autodoc2-docstring} src.envs.routing.vrpp.VRPPEnv._check_done
```

````

`````
