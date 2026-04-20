# {py:mod}`src.envs.routing.cvrp`

```{py:module} src.envs.routing.cvrp
```

```{autodoc2-docstring} src.envs.routing.cvrp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CVRPEnv <src.envs.routing.cvrp.CVRPEnv>`
  - ```{autodoc2-docstring} src.envs.routing.cvrp.CVRPEnv
    :summary:
    ```
````

### API

`````{py:class} CVRPEnv(generator: typing.Optional[logic.src.envs.generators.cvrp.CVRPGenerator] = None, generator_params: typing.Optional[dict] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.routing.cvrp.CVRPEnv

Bases: {py:obj}`logic.src.envs.base.base.RL4COEnvBase`

```{autodoc2-docstring} src.envs.routing.cvrp.CVRPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.routing.cvrp.CVRPEnv.__init__
```

````{py:attribute} NAME
:canonical: src.envs.routing.cvrp.CVRPEnv.NAME
:type: str
:value: >
   'cvrp'

```{autodoc2-docstring} src.envs.routing.cvrp.CVRPEnv.NAME
```

````

````{py:attribute} name
:canonical: src.envs.routing.cvrp.CVRPEnv.name
:type: str
:value: >
   'cvrp'

```{autodoc2-docstring} src.envs.routing.cvrp.CVRPEnv.name
```

````

````{py:attribute} node_dim
:canonical: src.envs.routing.cvrp.CVRPEnv.node_dim
:type: int
:value: >
   2

```{autodoc2-docstring} src.envs.routing.cvrp.CVRPEnv.node_dim
```

````

````{py:method} _reset_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.cvrp.CVRPEnv._reset_instance

```{autodoc2-docstring} src.envs.routing.cvrp.CVRPEnv._reset_instance
```

````

````{py:method} _step(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.cvrp.CVRPEnv._step

````

````{py:method} _step_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.cvrp.CVRPEnv._step_instance

```{autodoc2-docstring} src.envs.routing.cvrp.CVRPEnv._step_instance
```

````

````{py:method} _check_done(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.cvrp.CVRPEnv._check_done

```{autodoc2-docstring} src.envs.routing.cvrp.CVRPEnv._check_done
```

````

````{py:method} _get_action_mask(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.cvrp.CVRPEnv._get_action_mask

```{autodoc2-docstring} src.envs.routing.cvrp.CVRPEnv._get_action_mask
```

````

````{py:method} _get_reward(td: tensordict.TensorDictBase, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.routing.cvrp.CVRPEnv._get_reward

```{autodoc2-docstring} src.envs.routing.cvrp.CVRPEnv._get_reward
```

````

`````
