# {py:mod}`src.envs.routing.thop`

```{py:module} src.envs.routing.thop
```

```{autodoc2-docstring} src.envs.routing.thop
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ThOPEnv <src.envs.routing.thop.ThOPEnv>`
  - ```{autodoc2-docstring} src.envs.routing.thop.ThOPEnv
    :summary:
    ```
````

### API

`````{py:class} ThOPEnv(generator: typing.Optional[logic.src.envs.generators.thop.ThOPGenerator] = None, generator_params: typing.Optional[dict] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.routing.thop.ThOPEnv

Bases: {py:obj}`logic.src.envs.base.base.RL4COEnvBase`

```{autodoc2-docstring} src.envs.routing.thop.ThOPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.routing.thop.ThOPEnv.__init__
```

````{py:attribute} NAME
:canonical: src.envs.routing.thop.ThOPEnv.NAME
:type: str
:value: >
   'thop'

```{autodoc2-docstring} src.envs.routing.thop.ThOPEnv.NAME
```

````

````{py:attribute} name
:canonical: src.envs.routing.thop.ThOPEnv.name
:type: str
:value: >
   'thop'

```{autodoc2-docstring} src.envs.routing.thop.ThOPEnv.name
```

````

````{py:attribute} node_dim
:canonical: src.envs.routing.thop.ThOPEnv.node_dim
:type: int
:value: >
   4

```{autodoc2-docstring} src.envs.routing.thop.ThOPEnv.node_dim
```

````

````{py:method} _reset_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.thop.ThOPEnv._reset_instance

```{autodoc2-docstring} src.envs.routing.thop.ThOPEnv._reset_instance
```

````

````{py:method} _step(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.thop.ThOPEnv._step

```{autodoc2-docstring} src.envs.routing.thop.ThOPEnv._step
```

````

````{py:method} _step_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.thop.ThOPEnv._step_instance

```{autodoc2-docstring} src.envs.routing.thop.ThOPEnv._step_instance
```

````

````{py:method} _check_done(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.thop.ThOPEnv._check_done

```{autodoc2-docstring} src.envs.routing.thop.ThOPEnv._check_done
```

````

````{py:method} _get_action_mask(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.thop.ThOPEnv._get_action_mask

```{autodoc2-docstring} src.envs.routing.thop.ThOPEnv._get_action_mask
```

````

````{py:method} _get_reward(td: tensordict.TensorDictBase, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.routing.thop.ThOPEnv._get_reward

```{autodoc2-docstring} src.envs.routing.thop.ThOPEnv._get_reward
```

````

`````
