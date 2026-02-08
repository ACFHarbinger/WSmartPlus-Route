# {py:mod}`src.envs.tsp`

```{py:module} src.envs.tsp
```

```{autodoc2-docstring} src.envs.tsp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TSPEnv <src.envs.tsp.TSPEnv>`
  - ```{autodoc2-docstring} src.envs.tsp.TSPEnv
    :summary:
    ```
````

### API

`````{py:class} TSPEnv(generator: typing.Optional[logic.src.envs.generators.TSPGenerator] = None, generator_params: typing.Optional[dict] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.tsp.TSPEnv

Bases: {py:obj}`logic.src.envs.base.RL4COEnvBase`

```{autodoc2-docstring} src.envs.tsp.TSPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.tsp.TSPEnv.__init__
```

````{py:attribute} NAME
:canonical: src.envs.tsp.TSPEnv.NAME
:value: >
   'tsp'

```{autodoc2-docstring} src.envs.tsp.TSPEnv.NAME
```

````

````{py:attribute} name
:canonical: src.envs.tsp.TSPEnv.name
:type: str
:value: >
   'tsp'

```{autodoc2-docstring} src.envs.tsp.TSPEnv.name
```

````

````{py:attribute} node_dim
:canonical: src.envs.tsp.TSPEnv.node_dim
:type: int
:value: >
   2

```{autodoc2-docstring} src.envs.tsp.TSPEnv.node_dim
```

````

````{py:method} _reset_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.tsp.TSPEnv._reset_instance

```{autodoc2-docstring} src.envs.tsp.TSPEnv._reset_instance
```

````

````{py:method} _step_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.tsp.TSPEnv._step_instance

```{autodoc2-docstring} src.envs.tsp.TSPEnv._step_instance
```

````

````{py:method} _get_action_mask(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.tsp.TSPEnv._get_action_mask

```{autodoc2-docstring} src.envs.tsp.TSPEnv._get_action_mask
```

````

````{py:method} _get_reward(td: tensordict.TensorDict, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.tsp.TSPEnv._get_reward

```{autodoc2-docstring} src.envs.tsp.TSPEnv._get_reward
```

````

````{py:method} _check_done(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.tsp.TSPEnv._check_done

```{autodoc2-docstring} src.envs.tsp.TSPEnv._check_done
```

````

`````
