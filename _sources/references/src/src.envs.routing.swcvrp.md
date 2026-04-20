# {py:mod}`src.envs.routing.swcvrp`

```{py:module} src.envs.routing.swcvrp
```

```{autodoc2-docstring} src.envs.routing.swcvrp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SCWCVRPEnv <src.envs.routing.swcvrp.SCWCVRPEnv>`
  - ```{autodoc2-docstring} src.envs.routing.swcvrp.SCWCVRPEnv
    :summary:
    ```
````

### API

`````{py:class} SCWCVRPEnv(generator: typing.Optional[logic.src.envs.generators.SCWCVRPGenerator] = None, generator_params: typing.Optional[dict] = None, overflow_penalty: float = 10.0, waste_weight: float = 1.0, cost_weight: float = 1.0, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.routing.swcvrp.SCWCVRPEnv

Bases: {py:obj}`logic.src.envs.routing.wcvrp.WCVRPEnv`

```{autodoc2-docstring} src.envs.routing.swcvrp.SCWCVRPEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.routing.swcvrp.SCWCVRPEnv.__init__
```

````{py:attribute} name
:canonical: src.envs.routing.swcvrp.SCWCVRPEnv.name
:type: str
:value: >
   'scwcvrp'

```{autodoc2-docstring} src.envs.routing.swcvrp.SCWCVRPEnv.name
```

````

````{py:method} _reset_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.swcvrp.SCWCVRPEnv._reset_instance

```{autodoc2-docstring} src.envs.routing.swcvrp.SCWCVRPEnv._reset_instance
```

````

````{py:method} _step_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.routing.swcvrp.SCWCVRPEnv._step_instance

```{autodoc2-docstring} src.envs.routing.swcvrp.SCWCVRPEnv._step_instance
```

````

````{py:method} _get_reward(td: tensordict.TensorDictBase, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.routing.swcvrp.SCWCVRPEnv._get_reward

```{autodoc2-docstring} src.envs.routing.swcvrp.SCWCVRPEnv._get_reward
```

````

`````
