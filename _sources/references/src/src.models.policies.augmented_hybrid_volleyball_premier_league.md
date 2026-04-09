# {py:mod}`src.models.policies.augmented_hybrid_volleyball_premier_league`

```{py:module} src.models.policies.augmented_hybrid_volleyball_premier_league
```

```{autodoc2-docstring} src.models.policies.augmented_hybrid_volleyball_premier_league
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedAHVPL <src.models.policies.augmented_hybrid_volleyball_premier_league.VectorizedAHVPL>`
  - ```{autodoc2-docstring} src.models.policies.augmented_hybrid_volleyball_premier_league.VectorizedAHVPL
    :summary:
    ```
````

### API

`````{py:class} VectorizedAHVPL(env_name: str, n_teams: int = 10, max_iterations: int = 20, sub_rate: float = 0.2, time_limit: float = 60.0, aco_iterations: int = 1, alns_iterations: int = 50, crossover_rate: float = 0.7, elite_size: int = 5, device: str = 'cuda', generator: typing.Optional[torch.Generator] = None, rng: typing.Optional[random.Random] = None, **kwargs)
:canonical: src.models.policies.augmented_hybrid_volleyball_premier_league.VectorizedAHVPL

Bases: {py:obj}`logic.src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL`

```{autodoc2-docstring} src.models.policies.augmented_hybrid_volleyball_premier_league.VectorizedAHVPL
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.augmented_hybrid_volleyball_premier_league.VectorizedAHVPL.__init__
```

````{py:method} __getstate__() -> typing.Dict[str, typing.Any]
:canonical: src.models.policies.augmented_hybrid_volleyball_premier_league.VectorizedAHVPL.__getstate__

```{autodoc2-docstring} src.models.policies.augmented_hybrid_volleyball_premier_league.VectorizedAHVPL.__getstate__
```

````

````{py:method} __setstate__(state: typing.Dict[str, typing.Any]) -> None
:canonical: src.models.policies.augmented_hybrid_volleyball_premier_league.VectorizedAHVPL.__setstate__

```{autodoc2-docstring} src.models.policies.augmented_hybrid_volleyball_premier_league.VectorizedAHVPL.__setstate__
```

````

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, strategy: str = 'greedy', num_starts: int = 1, max_steps: typing.Optional[int] = None, phase: str = 'train', return_actions: bool = True, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.policies.augmented_hybrid_volleyball_premier_league.VectorizedAHVPL.forward

```{autodoc2-docstring} src.models.policies.augmented_hybrid_volleyball_premier_league.VectorizedAHVPL.forward
```

````

````{py:method} _ahvpl_substitution(pop_manager: logic.src.models.policies.hgs_core.population.VectorizedPopulation, dist_matrix: torch.Tensor, waste: torch.Tensor, capacity: torch.Tensor, tau: torch.Tensor, eta: torch.Tensor) -> None
:canonical: src.models.policies.augmented_hybrid_volleyball_premier_league.VectorizedAHVPL._ahvpl_substitution

```{autodoc2-docstring} src.models.policies.augmented_hybrid_volleyball_premier_league.VectorizedAHVPL._ahvpl_substitution
```

````

````{py:method} _routes_to_giant_tours(routes_list, batch_size, n_teams, N, backup_giant)
:canonical: src.models.policies.augmented_hybrid_volleyball_premier_league.VectorizedAHVPL._routes_to_giant_tours

```{autodoc2-docstring} src.models.policies.augmented_hybrid_volleyball_premier_league.VectorizedAHVPL._routes_to_giant_tours
```

````

`````
