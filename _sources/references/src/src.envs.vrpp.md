# {py:mod}`src.envs.vrpp`

```{py:module} src.envs.vrpp
```

```{autodoc2-docstring} src.envs.vrpp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VRPPEnv <src.envs.vrpp.VRPPEnv>`
  - ```{autodoc2-docstring} src.envs.vrpp.VRPPEnv
    :summary:
    ```
````

### API

`````{py:class} VRPPEnv(generator: typing.Optional[logic.src.envs.generators.VRPPGenerator] = None, generator_params: typing.Optional[dict] = None, prize_weight: float = 1.0, cost_weight: float = 1.0, revenue_kg: typing.Optional[float] = None, cost_km: typing.Optional[float] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.vrpp.VRPPEnv

Bases: {py:obj}`logic.src.envs.base.RL4COEnvBase`

```{autodoc2-docstring} src.envs.vrpp.VRPPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.vrpp.VRPPEnv.__init__
```

````{py:attribute} name
:canonical: src.envs.vrpp.VRPPEnv.name
:type: str
:value: >
   'vrpp'

```{autodoc2-docstring} src.envs.vrpp.VRPPEnv.name
```

````

````{py:method} _reset_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.vrpp.VRPPEnv._reset_instance

```{autodoc2-docstring} src.envs.vrpp.VRPPEnv._reset_instance
```

````

````{py:method} _step_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.vrpp.VRPPEnv._step_instance

```{autodoc2-docstring} src.envs.vrpp.VRPPEnv._step_instance
```

````

````{py:method} _get_action_mask(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.vrpp.VRPPEnv._get_action_mask

```{autodoc2-docstring} src.envs.vrpp.VRPPEnv._get_action_mask
```

````

````{py:method} _get_reward(td: tensordict.TensorDict, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.vrpp.VRPPEnv._get_reward

```{autodoc2-docstring} src.envs.vrpp.VRPPEnv._get_reward
```

````

````{py:method} _check_done(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.vrpp.VRPPEnv._check_done

```{autodoc2-docstring} src.envs.vrpp.VRPPEnv._check_done
```

````

`````
