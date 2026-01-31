# {py:mod}`src.models.policies.classical.alns`

```{py:module} src.models.policies.classical.alns
```

```{autodoc2-docstring} src.models.policies.classical.alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedALNS <src.models.policies.classical.alns.VectorizedALNS>`
  - ```{autodoc2-docstring} src.models.policies.classical.alns.VectorizedALNS
    :summary:
    ```
````

### API

`````{py:class} VectorizedALNS(env_name: str, time_limit: float = 5.0, max_iterations: int = 500, max_vehicles: int = 0, **kwargs)
:canonical: src.models.policies.classical.alns.VectorizedALNS

Bases: {py:obj}`logic.src.models.policies.base.ConstructivePolicy`

```{autodoc2-docstring} src.models.policies.classical.alns.VectorizedALNS
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.classical.alns.VectorizedALNS.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'greedy', num_starts: int = 1, **kwargs) -> dict
:canonical: src.models.policies.classical.alns.VectorizedALNS.forward

```{autodoc2-docstring} src.models.policies.classical.alns.VectorizedALNS.forward
```

````

`````
