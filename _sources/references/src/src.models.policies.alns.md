# {py:mod}`src.models.policies.alns`

```{py:module} src.models.policies.alns
```

```{autodoc2-docstring} src.models.policies.alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedALNS <src.models.policies.alns.VectorizedALNS>`
  - ```{autodoc2-docstring} src.models.policies.alns.VectorizedALNS
    :summary:
    ```
````

### API

`````{py:class} VectorizedALNS(env_name: str, time_limit: float = 5.0, max_iterations: int = 500, max_vehicles: int = 0, **kwargs)
:canonical: src.models.policies.alns.VectorizedALNS

Bases: {py:obj}`logic.src.models.common.improvement_policy.ImprovementPolicy`

```{autodoc2-docstring} src.models.policies.alns.VectorizedALNS
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.alns.VectorizedALNS.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'greedy', num_starts: int = 1, phase: str = 'train', return_actions: bool = True, **kwargs) -> dict[str, typing.Any]
:canonical: src.models.policies.alns.VectorizedALNS.forward

```{autodoc2-docstring} src.models.policies.alns.VectorizedALNS.forward
```

````

`````
