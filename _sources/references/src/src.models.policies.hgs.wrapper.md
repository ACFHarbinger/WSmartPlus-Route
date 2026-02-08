# {py:mod}`src.models.policies.hgs.wrapper`

```{py:module} src.models.policies.hgs.wrapper
```

```{autodoc2-docstring} src.models.policies.hgs.wrapper
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedHGS <src.models.policies.hgs.wrapper.VectorizedHGS>`
  - ```{autodoc2-docstring} src.models.policies.hgs.wrapper.VectorizedHGS
    :summary:
    ```
````

### API

`````{py:class} VectorizedHGS(env_name: str | None, time_limit: float = 5.0, population_size: int = 50, n_generations: int = 50, elite_size: int = 5, max_vehicles: int = 0, **kwargs)
:canonical: src.models.policies.hgs.wrapper.VectorizedHGS

Bases: {py:obj}`logic.src.models.common.improvement_policy.ImprovementPolicy`

```{autodoc2-docstring} src.models.policies.hgs.wrapper.VectorizedHGS
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.hgs.wrapper.VectorizedHGS.__init__
```

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'greedy', num_starts: int = 1, phase: str = 'train', return_actions: bool = True, **kwargs) -> dict[str, typing.Any]
:canonical: src.models.policies.hgs.wrapper.VectorizedHGS.forward

```{autodoc2-docstring} src.models.policies.hgs.wrapper.VectorizedHGS.forward
```

````

`````
