# {py:mod}`src.envs.jssp`

```{py:module} src.envs.jssp
```

```{autodoc2-docstring} src.envs.jssp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JSSPEnv <src.envs.jssp.JSSPEnv>`
  - ```{autodoc2-docstring} src.envs.jssp.JSSPEnv
    :summary:
    ```
````

### API

`````{py:class} JSSPEnv(generator: typing.Optional[logic.src.envs.generators.JSSPGenerator] = None, generator_params: typing.Optional[dict] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.jssp.JSSPEnv

Bases: {py:obj}`logic.src.envs.base.RL4COEnvBase`

```{autodoc2-docstring} src.envs.jssp.JSSPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.jssp.JSSPEnv.__init__
```

````{py:attribute} name
:canonical: src.envs.jssp.JSSPEnv.name
:type: str
:value: >
   'jssp'

```{autodoc2-docstring} src.envs.jssp.JSSPEnv.name
```

````

````{py:method} _reset_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.jssp.JSSPEnv._reset_instance

```{autodoc2-docstring} src.envs.jssp.JSSPEnv._reset_instance
```

````

````{py:method} _step_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.jssp.JSSPEnv._step_instance

```{autodoc2-docstring} src.envs.jssp.JSSPEnv._step_instance
```

````

````{py:method} _get_action_mask(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.jssp.JSSPEnv._get_action_mask

```{autodoc2-docstring} src.envs.jssp.JSSPEnv._get_action_mask
```

````

````{py:method} _get_reward(td: tensordict.TensorDict, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.jssp.JSSPEnv._get_reward

```{autodoc2-docstring} src.envs.jssp.JSSPEnv._get_reward
```

````

````{py:method} _check_done(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.jssp.JSSPEnv._check_done

```{autodoc2-docstring} src.envs.jssp.JSSPEnv._check_done
```

````

`````
