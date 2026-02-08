# {py:mod}`src.envs.tsp_kopt`

```{py:module} src.envs.tsp_kopt
```

```{autodoc2-docstring} src.envs.tsp_kopt
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TSPkoptEnv <src.envs.tsp_kopt.TSPkoptEnv>`
  - ```{autodoc2-docstring} src.envs.tsp_kopt.TSPkoptEnv
    :summary:
    ```
````

### API

`````{py:class} TSPkoptEnv(generator: typing.Optional[logic.src.envs.generators.TSPGenerator] = None, generator_params: typing.Optional[dict] = None, device: typing.Union[str, torch.device] = 'cpu', **kwargs)
:canonical: src.envs.tsp_kopt.TSPkoptEnv

Bases: {py:obj}`logic.src.envs.base.ImprovementEnvBase`

```{autodoc2-docstring} src.envs.tsp_kopt.TSPkoptEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.tsp_kopt.TSPkoptEnv.__init__
```

````{py:attribute} name
:canonical: src.envs.tsp_kopt.TSPkoptEnv.name
:type: str
:value: >
   'tsp_kopt'

```{autodoc2-docstring} src.envs.tsp_kopt.TSPkoptEnv.name
```

````

````{py:method} _get_action_mask(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.tsp_kopt.TSPkoptEnv._get_action_mask

```{autodoc2-docstring} src.envs.tsp_kopt.TSPkoptEnv._get_action_mask
```

````

````{py:method} _get_initial_solution(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.tsp_kopt.TSPkoptEnv._get_initial_solution

```{autodoc2-docstring} src.envs.tsp_kopt.TSPkoptEnv._get_initial_solution
```

````

````{py:method} _step_instance(td: tensordict.TensorDict) -> tensordict.TensorDict
:canonical: src.envs.tsp_kopt.TSPkoptEnv._step_instance

```{autodoc2-docstring} src.envs.tsp_kopt.TSPkoptEnv._step_instance
```

````

````{py:method} _get_reward(td: tensordict.TensorDict, actions: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.envs.tsp_kopt.TSPkoptEnv._get_reward

```{autodoc2-docstring} src.envs.tsp_kopt.TSPkoptEnv._get_reward
```

````

````{py:method} _get_initial_reward(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.tsp_kopt.TSPkoptEnv._get_initial_reward

```{autodoc2-docstring} src.envs.tsp_kopt.TSPkoptEnv._get_initial_reward
```

````

````{py:method} _check_done(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.envs.tsp_kopt.TSPkoptEnv._check_done

```{autodoc2-docstring} src.envs.tsp_kopt.TSPkoptEnv._check_done
```

````

`````
