# {py:mod}`src.envs.pdp`

```{py:module} src.envs.pdp
```

```{autodoc2-docstring} src.envs.pdp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PDPEnv <src.envs.pdp.PDPEnv>`
  - ```{autodoc2-docstring} src.envs.pdp.PDPEnv
    :summary:
    ```
````

### API

`````{py:class} PDPEnv(generator: typing.Optional[logic.src.envs.generators.PDPGenerator] = None, generator_params: typing.Optional[dict] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.pdp.PDPEnv

Bases: {py:obj}`logic.src.envs.base.RL4COEnvBase`

```{autodoc2-docstring} src.envs.pdp.PDPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.pdp.PDPEnv.__init__
```

````{py:attribute} name
:canonical: src.envs.pdp.PDPEnv.name
:type: str
:value: >
   'pdp'

```{autodoc2-docstring} src.envs.pdp.PDPEnv.name
```

````

````{py:method} _reset_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.pdp.PDPEnv._reset_instance

```{autodoc2-docstring} src.envs.pdp.PDPEnv._reset_instance
```

````

````{py:method} _step_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.pdp.PDPEnv._step_instance

```{autodoc2-docstring} src.envs.pdp.PDPEnv._step_instance
```

````

````{py:method} _get_action_mask(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.pdp.PDPEnv._get_action_mask

```{autodoc2-docstring} src.envs.pdp.PDPEnv._get_action_mask
```

````

````{py:method} _get_reward(td: tensordict.TensorDict, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.pdp.PDPEnv._get_reward

```{autodoc2-docstring} src.envs.pdp.PDPEnv._get_reward
```

````

````{py:method} _check_done(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.pdp.PDPEnv._check_done

```{autodoc2-docstring} src.envs.pdp.PDPEnv._check_done
```

````

`````
