# {py:mod}`src.policies.vector.alns`

```{py:module} src.policies.vector.alns
```

```{autodoc2-docstring} src.policies.vector.alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedALNS <src.policies.vector.alns.VectorizedALNS>`
  - ```{autodoc2-docstring} src.policies.vector.alns.VectorizedALNS
    :summary:
    ```
````

### API

`````{py:class} VectorizedALNS(env_name: str, time_limit: float = 5.0, max_iterations: int = 500, max_vehicles: int = 0, start_temp: float = 0.5, cooling_rate: float = 0.9995, device: str = 'cpu', seed: int = 42, **kwargs: typing.Any)
:canonical: src.policies.vector.alns.VectorizedALNS

Bases: {py:obj}`logic.src.models.common.autoregressive.policy.AutoregressivePolicy`

```{autodoc2-docstring} src.policies.vector.alns.VectorizedALNS
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.vector.alns.VectorizedALNS.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.base.RL4COEnvBase] = None, strategy: str = 'greedy', num_starts: int = 1, max_steps: typing.Optional[int] = None, phase: str = 'train', return_actions: bool = True, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.policies.vector.alns.VectorizedALNS.forward

```{autodoc2-docstring} src.policies.vector.alns.VectorizedALNS.forward
```

````

`````
