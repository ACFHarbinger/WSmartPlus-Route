# {py:mod}`src.envs.routing.irp`

```{py:module} src.envs.routing.irp
```

```{autodoc2-docstring} src.envs.routing.irp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IRPEnv <src.envs.routing.irp.IRPEnv>`
  - ```{autodoc2-docstring} src.envs.routing.irp.IRPEnv
    :summary:
    ```
````

### API

`````{py:class} IRPEnv(generator: typing.Optional[logic.src.envs.generators.irp.IRPGenerator] = None, generator_params: typing.Optional[dict] = None, stockout_penalty: float = 10.0, holding_cost_weight: float = 1.0, routing_cost_weight: float = 1.0, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.routing.irp.IRPEnv

Bases: {py:obj}`logic.src.envs.base.base.RL4COEnvBase`

```{autodoc2-docstring} src.envs.routing.irp.IRPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.routing.irp.IRPEnv.__init__
```

````{py:attribute} NAME
:canonical: src.envs.routing.irp.IRPEnv.NAME
:type: str
:value: >
   'irp'

```{autodoc2-docstring} src.envs.routing.irp.IRPEnv.NAME
```

````

````{py:attribute} name
:canonical: src.envs.routing.irp.IRPEnv.name
:type: str
:value: >
   'irp'

```{autodoc2-docstring} src.envs.routing.irp.IRPEnv.name
```

````

````{py:attribute} node_dim
:canonical: src.envs.routing.irp.IRPEnv.node_dim
:type: int
:value: >
   2

```{autodoc2-docstring} src.envs.routing.irp.IRPEnv.node_dim
```

````

````{py:method} _reset_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.irp.IRPEnv._reset_instance

```{autodoc2-docstring} src.envs.routing.irp.IRPEnv._reset_instance
```

````

````{py:method} _step(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.irp.IRPEnv._step

````

````{py:method} _step_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.irp.IRPEnv._step_instance

```{autodoc2-docstring} src.envs.routing.irp.IRPEnv._step_instance
```

````

````{py:method} _check_done(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.irp.IRPEnv._check_done

```{autodoc2-docstring} src.envs.routing.irp.IRPEnv._check_done
```

````

````{py:method} _get_action_mask(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.irp.IRPEnv._get_action_mask

```{autodoc2-docstring} src.envs.routing.irp.IRPEnv._get_action_mask
```

````

````{py:method} _get_reward(td: tensordict.TensorDictBase, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.routing.irp.IRPEnv._get_reward

```{autodoc2-docstring} src.envs.routing.irp.IRPEnv._get_reward
```

````

`````
