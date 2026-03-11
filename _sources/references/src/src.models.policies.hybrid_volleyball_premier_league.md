# {py:mod}`src.models.policies.hybrid_volleyball_premier_league`

```{py:module} src.models.policies.hybrid_volleyball_premier_league
```

```{autodoc2-docstring} src.models.policies.hybrid_volleyball_premier_league
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedHVPL <src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL>`
  - ```{autodoc2-docstring} src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL
    :summary:
    ```
````

### API

`````{py:class} VectorizedHVPL(env_name: str, n_teams: int = 10, max_iterations: int = 20, sub_rate: float = 0.2, time_limit: float = 60.0, aco_iterations: int = 1, alns_iterations: int = 100, device: str = 'cuda', generator: typing.Optional[torch.Generator] = None, rng: typing.Optional[random.Random] = None, **kwargs)
:canonical: src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL

Bases: {py:obj}`logic.src.models.common.autoregressive.policy.AutoregressivePolicy`

```{autodoc2-docstring} src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL.__init__
```

````{py:method} __getstate__() -> typing.Dict[str, typing.Any]
:canonical: src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL.__getstate__

```{autodoc2-docstring} src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL.__getstate__
```

````

````{py:method} __setstate__(state: typing.Dict[str, typing.Any]) -> None
:canonical: src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL.__setstate__

```{autodoc2-docstring} src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL.__setstate__
```

````

````{py:method} forward(td: tensordict.TensorDict, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None, strategy: str = 'greedy', num_starts: int = 1, max_steps: typing.Optional[int] = None, phase: str = 'train', return_actions: bool = True, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL.forward

```{autodoc2-docstring} src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL.forward
```

````

````{py:method} _setup_data(td: tensordict.TensorDict) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL._setup_data

```{autodoc2-docstring} src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL._setup_data
```

````

````{py:method} _coaching_phase(population_tours: torch.Tensor, dist: torch.Tensor, waste: torch.Tensor, capacity: torch.Tensor) -> typing.Tuple[torch.Tensor, typing.List[typing.List[int]]]
:canonical: src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL._coaching_phase

```{autodoc2-docstring} src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL._coaching_phase
```

````

````{py:method} _global_competition(instance_costs: torch.Tensor, coached_routes_list: typing.List[typing.List[int]], best_tours: torch.Tensor, best_costs: torch.Tensor, num_nodes: int) -> torch.Tensor
:canonical: src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL._global_competition

```{autodoc2-docstring} src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL._global_competition
```

````

````{py:method} _substitution_phase(population_tours: torch.Tensor, instance_costs: torch.Tensor, dist_matrix: torch.Tensor, tau: torch.Tensor, eta: torch.Tensor) -> None
:canonical: src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL._substitution_phase

```{autodoc2-docstring} src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL._substitution_phase
```

````

````{py:method} _format_output(td: tensordict.TensorDict, best_tours: torch.Tensor, dist_matrix: torch.Tensor, waste: torch.Tensor, capacity: torch.Tensor, env: typing.Optional[logic.src.envs.base.RL4COEnvBase] = None) -> typing.Dict[str, typing.Any]
:canonical: src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL._format_output

```{autodoc2-docstring} src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL._format_output
```

````

````{py:method} _aco_construct(dist_matrix: torch.Tensor, tau: torch.Tensor, eta: torch.Tensor, n_ants_per_instance: int) -> torch.Tensor
:canonical: src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL._aco_construct

```{autodoc2-docstring} src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL._aco_construct
```

````

````{py:method} _update_pheromones(tau: torch.Tensor, best_tours: torch.Tensor, best_costs: torch.Tensor) -> torch.Tensor
:canonical: src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL._update_pheromones

```{autodoc2-docstring} src.models.policies.hybrid_volleyball_premier_league.VectorizedHVPL._update_pheromones
```

````

`````
