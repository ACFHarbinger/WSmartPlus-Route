# {py:mod}`src.models.policies.hgs_alns`

```{py:module} src.models.policies.hgs_alns
```

```{autodoc2-docstring} src.models.policies.hgs_alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedHGSALNS <src.models.policies.hgs_alns.VectorizedHGSALNS>`
  - ```{autodoc2-docstring} src.models.policies.hgs_alns.VectorizedHGSALNS
    :summary:
    ```
````

### API

`````{py:class} VectorizedHGSALNS(env_name: str | None = 'vrpp', time_limit: float = 0.1, population_size: int = 20, n_generations: int = 15, elite_size: int = 2, max_vehicles: int = 0, alns_education_iterations: int = 5, **kwargs: typing.Any)
:canonical: src.models.policies.hgs_alns.VectorizedHGSALNS

Bases: {py:obj}`logic.src.models.policies.hgs.VectorizedHGS`

```{autodoc2-docstring} src.models.policies.hgs_alns.VectorizedHGSALNS
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.hgs_alns.VectorizedHGSALNS.__init__
```

````{py:method} solve(dist_matrix, demands, capacity, **kwargs)
:canonical: src.models.policies.hgs_alns.VectorizedHGSALNS.solve

```{autodoc2-docstring} src.models.policies.hgs_alns.VectorizedHGSALNS.solve
```

````

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'greedy', num_starts: int = 1, **kwargs) -> dict
:canonical: src.models.policies.hgs_alns.VectorizedHGSALNS.forward

```{autodoc2-docstring} src.models.policies.hgs_alns.VectorizedHGSALNS.forward
```

````

`````
