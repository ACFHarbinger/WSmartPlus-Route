# {py:mod}`src.envs.swcvrp`

```{py:module} src.envs.swcvrp
```

```{autodoc2-docstring} src.envs.swcvrp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SCWCVRPEnv <src.envs.swcvrp.SCWCVRPEnv>`
  - ```{autodoc2-docstring} src.envs.swcvrp.SCWCVRPEnv
    :summary:
    ```
````

### API

`````{py:class} SCWCVRPEnv(generator: typing.Optional[logic.src.envs.generators.SCWCVRPGenerator] = None, generator_params: typing.Optional[dict] = None, overflow_penalty: float = 10.0, collection_reward: float = 1.0, cost_weight: float = 1.0, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.swcvrp.SCWCVRPEnv

Bases: {py:obj}`logic.src.envs.wcvrp.WCVRPEnv`

```{autodoc2-docstring} src.envs.swcvrp.SCWCVRPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.swcvrp.SCWCVRPEnv.__init__
```

````{py:attribute} name
:canonical: src.envs.swcvrp.SCWCVRPEnv.name
:type: str
:value: >
   'scwcvrp'

```{autodoc2-docstring} src.envs.swcvrp.SCWCVRPEnv.name
```

````

````{py:method} _reset_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.swcvrp.SCWCVRPEnv._reset_instance

```{autodoc2-docstring} src.envs.swcvrp.SCWCVRPEnv._reset_instance
```

````

````{py:method} _step_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.swcvrp.SCWCVRPEnv._step_instance

```{autodoc2-docstring} src.envs.swcvrp.SCWCVRPEnv._step_instance
```

````

````{py:method} _get_reward(td: tensordict.TensorDict, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.swcvrp.SCWCVRPEnv._get_reward

```{autodoc2-docstring} src.envs.swcvrp.SCWCVRPEnv._get_reward
```

````

`````
