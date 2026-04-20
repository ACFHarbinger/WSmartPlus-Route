# {py:mod}`src.envs.routing.atsp`

```{py:module} src.envs.routing.atsp
```

```{autodoc2-docstring} src.envs.routing.atsp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ATSPEnv <src.envs.routing.atsp.ATSPEnv>`
  - ```{autodoc2-docstring} src.envs.routing.atsp.ATSPEnv
    :summary:
    ```
````

### API

`````{py:class} ATSPEnv(generator: typing.Optional[logic.src.envs.generators.atsp.ATSPGenerator] = None, generator_params: typing.Optional[dict] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.routing.atsp.ATSPEnv

Bases: {py:obj}`logic.src.envs.base.base.RL4COEnvBase`

```{autodoc2-docstring} src.envs.routing.atsp.ATSPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.routing.atsp.ATSPEnv.__init__
```

````{py:attribute} NAME
:canonical: src.envs.routing.atsp.ATSPEnv.NAME
:type: str
:value: >
   'atsp'

```{autodoc2-docstring} src.envs.routing.atsp.ATSPEnv.NAME
```

````

````{py:attribute} name
:canonical: src.envs.routing.atsp.ATSPEnv.name
:type: str
:value: >
   'atsp'

```{autodoc2-docstring} src.envs.routing.atsp.ATSPEnv.name
```

````

````{py:attribute} node_dim
:canonical: src.envs.routing.atsp.ATSPEnv.node_dim
:type: int
:value: >
   0

```{autodoc2-docstring} src.envs.routing.atsp.ATSPEnv.node_dim
```

````

````{py:method} _reset_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.atsp.ATSPEnv._reset_instance

```{autodoc2-docstring} src.envs.routing.atsp.ATSPEnv._reset_instance
```

````

````{py:method} _step(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.atsp.ATSPEnv._step

````

````{py:method} _step_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.atsp.ATSPEnv._step_instance

```{autodoc2-docstring} src.envs.routing.atsp.ATSPEnv._step_instance
```

````

````{py:method} _check_done(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.atsp.ATSPEnv._check_done

```{autodoc2-docstring} src.envs.routing.atsp.ATSPEnv._check_done
```

````

````{py:method} _get_action_mask(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.atsp.ATSPEnv._get_action_mask

```{autodoc2-docstring} src.envs.routing.atsp.ATSPEnv._get_action_mask
```

````

````{py:method} _get_reward(td: tensordict.TensorDictBase, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.routing.atsp.ATSPEnv._get_reward

```{autodoc2-docstring} src.envs.routing.atsp.ATSPEnv._get_reward
```

````

`````
