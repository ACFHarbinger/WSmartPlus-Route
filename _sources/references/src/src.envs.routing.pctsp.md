# {py:mod}`src.envs.routing.pctsp`

```{py:module} src.envs.routing.pctsp
```

```{autodoc2-docstring} src.envs.routing.pctsp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PCTSPEnv <src.envs.routing.pctsp.PCTSPEnv>`
  - ```{autodoc2-docstring} src.envs.routing.pctsp.PCTSPEnv
    :summary:
    ```
````

### API

`````{py:class} PCTSPEnv(generator: typing.Optional[logic.src.envs.generators.pctsp.PCTSPGenerator] = None, generator_params: typing.Optional[dict] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.routing.pctsp.PCTSPEnv

Bases: {py:obj}`logic.src.envs.base.base.RL4COEnvBase`

```{autodoc2-docstring} src.envs.routing.pctsp.PCTSPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.routing.pctsp.PCTSPEnv.__init__
```

````{py:attribute} NAME
:canonical: src.envs.routing.pctsp.PCTSPEnv.NAME
:type: str
:value: >
   'pctsp'

```{autodoc2-docstring} src.envs.routing.pctsp.PCTSPEnv.NAME
```

````

````{py:attribute} name
:canonical: src.envs.routing.pctsp.PCTSPEnv.name
:type: str
:value: >
   'pctsp'

```{autodoc2-docstring} src.envs.routing.pctsp.PCTSPEnv.name
```

````

````{py:attribute} node_dim
:canonical: src.envs.routing.pctsp.PCTSPEnv.node_dim
:type: int
:value: >
   2

```{autodoc2-docstring} src.envs.routing.pctsp.PCTSPEnv.node_dim
```

````

````{py:attribute} _stochastic
:canonical: src.envs.routing.pctsp.PCTSPEnv._stochastic
:type: bool
:value: >
   False

```{autodoc2-docstring} src.envs.routing.pctsp.PCTSPEnv._stochastic
```

````

````{py:method} _reset_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.pctsp.PCTSPEnv._reset_instance

```{autodoc2-docstring} src.envs.routing.pctsp.PCTSPEnv._reset_instance
```

````

````{py:method} _step(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.pctsp.PCTSPEnv._step

````

````{py:method} _step_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.pctsp.PCTSPEnv._step_instance

```{autodoc2-docstring} src.envs.routing.pctsp.PCTSPEnv._step_instance
```

````

````{py:method} _check_done(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.pctsp.PCTSPEnv._check_done

```{autodoc2-docstring} src.envs.routing.pctsp.PCTSPEnv._check_done
```

````

````{py:method} _get_action_mask(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.routing.pctsp.PCTSPEnv._get_action_mask

```{autodoc2-docstring} src.envs.routing.pctsp.PCTSPEnv._get_action_mask
```

````

````{py:method} _get_reward(td: tensordict.TensorDictBase, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.routing.pctsp.PCTSPEnv._get_reward

```{autodoc2-docstring} src.envs.routing.pctsp.PCTSPEnv._get_reward
```

````

`````
