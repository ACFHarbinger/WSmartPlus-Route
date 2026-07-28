# {py:mod}`src.envs.routing.op`

```{py:module} src.envs.routing.op
```

```{autodoc2-docstring} src.envs.routing.op
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OPEnv <src.envs.routing.op.OPEnv>`
  - ```{autodoc2-docstring} src.envs.routing.op.OPEnv
    :summary:
    ```
````

### API

`````{py:class} OPEnv(generator: typing.Optional[logic.src.envs.generators.op.OPGenerator] = None, generator_params: typing.Optional[dict] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.routing.op.OPEnv

Bases: {py:obj}`logic.src.envs.base.base.RL4COEnvBase`

```{autodoc2-docstring} src.envs.routing.op.OPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.routing.op.OPEnv.__init__
```

````{py:attribute} NAME
:canonical: src.envs.routing.op.OPEnv.NAME
:type: str
:value: >
   'op'

```{autodoc2-docstring} src.envs.routing.op.OPEnv.NAME
```

````

````{py:attribute} name
:canonical: src.envs.routing.op.OPEnv.name
:type: str
:value: >
   'op'

```{autodoc2-docstring} src.envs.routing.op.OPEnv.name
```

````

````{py:attribute} node_dim
:canonical: src.envs.routing.op.OPEnv.node_dim
:type: int
:value: >
   2

```{autodoc2-docstring} src.envs.routing.op.OPEnv.node_dim
```

````

````{py:method} _reset_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.op.OPEnv._reset_instance

```{autodoc2-docstring} src.envs.routing.op.OPEnv._reset_instance
```

````

````{py:method} _step(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.op.OPEnv._step

```{autodoc2-docstring} src.envs.routing.op.OPEnv._step
```

````

````{py:method} _step_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.op.OPEnv._step_instance

```{autodoc2-docstring} src.envs.routing.op.OPEnv._step_instance
```

````

````{py:method} _check_done(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.op.OPEnv._check_done

```{autodoc2-docstring} src.envs.routing.op.OPEnv._check_done
```

````

````{py:method} _get_action_mask(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.op.OPEnv._get_action_mask

```{autodoc2-docstring} src.envs.routing.op.OPEnv._get_action_mask
```

````

````{py:method} _get_reward(td: tensordict.TensorDictBase, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.routing.op.OPEnv._get_reward

```{autodoc2-docstring} src.envs.routing.op.OPEnv._get_reward
```

````

`````
