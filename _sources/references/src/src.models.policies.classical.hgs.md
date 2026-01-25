# {py:mod}`src.models.policies.classical.hgs`

```{py:module} src.models.policies.classical.hgs
```

```{autodoc2-docstring} src.models.policies.classical.hgs
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSPolicy <src.models.policies.classical.hgs.HGSPolicy>`
  - ```{autodoc2-docstring} src.models.policies.classical.hgs.HGSPolicy
    :summary:
    ```
````

### API

`````{py:class} HGSPolicy(env_name: str, time_limit: float = 5.0, population_size: int = 50, n_generations: int = 50, elite_size: int = 5, max_vehicles: int = 0, **kwargs)
:canonical: src.models.policies.classical.hgs.HGSPolicy

Bases: {py:obj}`logic.src.models.policies.base.ConstructivePolicy`

```{autodoc2-docstring} src.models.policies.classical.hgs.HGSPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.classical.hgs.HGSPolicy.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'greedy', num_starts: int = 1, **kwargs) -> dict
:canonical: src.models.policies.classical.hgs.HGSPolicy.forward

```{autodoc2-docstring} src.models.policies.classical.hgs.HGSPolicy.forward
```

````

`````
