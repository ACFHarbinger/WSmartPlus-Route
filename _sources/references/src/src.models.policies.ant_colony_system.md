# {py:mod}`src.models.policies.ant_colony_system`

```{py:module} src.models.policies.ant_colony_system
```

```{autodoc2-docstring} src.models.policies.ant_colony_system
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedACOPolicy <src.models.policies.ant_colony_system.VectorizedACOPolicy>`
  - ```{autodoc2-docstring} src.models.policies.ant_colony_system.VectorizedACOPolicy
    :summary:
    ```
````

### API

`````{py:class} VectorizedACOPolicy(env_name: str, n_ants: int = 20, n_iterations: int = 50, alpha: float = 1.0, beta: float = 2.0, decay: float = 0.1, elitism: int = 1, q0: float = 0.9, min_pheromone: float = 0.01, **kwargs)
:canonical: src.models.policies.ant_colony_system.VectorizedACOPolicy

Bases: {py:obj}`logic.src.models.common.autoregressive_policy.AutoregressivePolicy`

```{autodoc2-docstring} src.models.policies.ant_colony_system.VectorizedACOPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.ant_colony_system.VectorizedACOPolicy.__init__
```

````{py:method} _get_heuristic(dist_matrix: torch.Tensor) -> torch.Tensor
:canonical: src.models.policies.ant_colony_system.VectorizedACOPolicy._get_heuristic

```{autodoc2-docstring} src.models.policies.ant_colony_system.VectorizedACOPolicy._get_heuristic
```

````

````{py:method} forward(td: tensordict.TensorDict, env: logic.src.envs.base.RL4COEnvBase, decode_type: str = 'sampling', num_starts: int = 1, **kwargs) -> typing.Dict[str, typing.Any]
:canonical: src.models.policies.ant_colony_system.VectorizedACOPolicy.forward

```{autodoc2-docstring} src.models.policies.ant_colony_system.VectorizedACOPolicy.forward
```

````

````{py:method} _construct_solutions(dist_matrix: torch.Tensor, tau: torch.Tensor, eta: torch.Tensor, env: typing.Optional[logic.src.envs.base.RL4COEnvBase]) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.policies.ant_colony_system.VectorizedACOPolicy._construct_solutions

```{autodoc2-docstring} src.models.policies.ant_colony_system.VectorizedACOPolicy._construct_solutions
```

````

````{py:method} _evaluate_batch(ant_tours: torch.Tensor, dist_matrix: torch.Tensor) -> torch.Tensor
:canonical: src.models.policies.ant_colony_system.VectorizedACOPolicy._evaluate_batch

```{autodoc2-docstring} src.models.policies.ant_colony_system.VectorizedACOPolicy._evaluate_batch
```

````

````{py:method} _update_pheromones(tau: torch.Tensor, ant_tours: torch.Tensor, costs: torch.Tensor) -> torch.Tensor
:canonical: src.models.policies.ant_colony_system.VectorizedACOPolicy._update_pheromones

```{autodoc2-docstring} src.models.policies.ant_colony_system.VectorizedACOPolicy._update_pheromones
```

````

`````
