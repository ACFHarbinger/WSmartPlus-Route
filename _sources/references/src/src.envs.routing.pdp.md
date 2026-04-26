# {py:mod}`src.envs.routing.pdp`

```{py:module} src.envs.routing.pdp
```

```{autodoc2-docstring} src.envs.routing.pdp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PDPEnv <src.envs.routing.pdp.PDPEnv>`
  - ```{autodoc2-docstring} src.envs.routing.pdp.PDPEnv
    :summary:
    ```
````

### API

`````{py:class} PDPEnv(generator: typing.Optional[logic.src.envs.generators.pdp.PDPGenerator] = None, generator_params: typing.Optional[dict] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.routing.pdp.PDPEnv

Bases: {py:obj}`logic.src.envs.base.base.RL4COEnvBase`

```{autodoc2-docstring} src.envs.routing.pdp.PDPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.routing.pdp.PDPEnv.__init__
```

````{py:attribute} NAME
:canonical: src.envs.routing.pdp.PDPEnv.NAME
:type: str
:value: >
   'pdp'

```{autodoc2-docstring} src.envs.routing.pdp.PDPEnv.NAME
```

````

````{py:attribute} name
:canonical: src.envs.routing.pdp.PDPEnv.name
:type: str
:value: >
   'pdp'

```{autodoc2-docstring} src.envs.routing.pdp.PDPEnv.name
```

````

````{py:attribute} node_dim
:canonical: src.envs.routing.pdp.PDPEnv.node_dim
:type: int
:value: >
   2

```{autodoc2-docstring} src.envs.routing.pdp.PDPEnv.node_dim
```

````

````{py:method} _reset_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.pdp.PDPEnv._reset_instance

```{autodoc2-docstring} src.envs.routing.pdp.PDPEnv._reset_instance
```

````

````{py:method} _step(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.pdp.PDPEnv._step

```{autodoc2-docstring} src.envs.routing.pdp.PDPEnv._step
```

````

````{py:method} _step_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.pdp.PDPEnv._step_instance

```{autodoc2-docstring} src.envs.routing.pdp.PDPEnv._step_instance
```

````

````{py:method} _check_done(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.pdp.PDPEnv._check_done

```{autodoc2-docstring} src.envs.routing.pdp.PDPEnv._check_done
```

````

````{py:method} _get_action_mask(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.pdp.PDPEnv._get_action_mask

```{autodoc2-docstring} src.envs.routing.pdp.PDPEnv._get_action_mask
```

````

````{py:method} _get_reward(td: tensordict.TensorDictBase, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.routing.pdp.PDPEnv._get_reward

```{autodoc2-docstring} src.envs.routing.pdp.PDPEnv._get_reward
```

````

`````
