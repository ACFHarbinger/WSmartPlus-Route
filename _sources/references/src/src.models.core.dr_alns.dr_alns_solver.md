# {py:mod}`src.models.core.dr_alns.dr_alns_solver`

```{py:module} src.models.core.dr_alns.dr_alns_solver
```

```{autodoc2-docstring} src.models.core.dr_alns.dr_alns_solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DRALNSSolver <src.models.core.dr_alns.dr_alns_solver.DRALNSSolver>`
  - ```{autodoc2-docstring} src.models.core.dr_alns.dr_alns_solver.DRALNSSolver
    :summary:
    ```
````

### API

`````{py:class} DRALNSSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, agent: src.models.core.dr_alns.ppo_agent.DRALNSPPOAgent, max_iterations: int = 100, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None, device: typing.Optional[torch.device] = None)
:canonical: src.models.core.dr_alns.dr_alns_solver.DRALNSSolver

```{autodoc2-docstring} src.models.core.dr_alns.dr_alns_solver.DRALNSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.dr_alns.dr_alns_solver.DRALNSSolver.__init__
```

````{py:method} solve(initial_routes: typing.Optional[typing.List[typing.List[int]]] = None) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.models.core.dr_alns.dr_alns_solver.DRALNSSolver.solve

```{autodoc2-docstring} src.models.core.dr_alns.dr_alns_solver.DRALNSSolver.solve
```

````

````{py:method} _apply_operators(routes: typing.List[typing.List[int]], destroy_idx: int, repair_idx: int, severity: float) -> typing.List[typing.List[int]]
:canonical: src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._apply_operators

```{autodoc2-docstring} src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._apply_operators
```

````

````{py:method} _accept_solution(current_profit: float, new_profit: float, temperature: float) -> bool
:canonical: src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._accept_solution

```{autodoc2-docstring} src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._accept_solution
```

````

````{py:method} _random_removal(routes: typing.List[typing.List[int]], n: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._random_removal

```{autodoc2-docstring} src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._random_removal
```

````

````{py:method} _worst_removal(routes: typing.List[typing.List[int]], n: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._worst_removal

```{autodoc2-docstring} src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._worst_removal
```

````

````{py:method} _cluster_removal(routes: typing.List[typing.List[int]], n: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._cluster_removal

```{autodoc2-docstring} src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._cluster_removal
```

````

````{py:method} _greedy_insertion(partial_routes: typing.List[typing.List[int]], removed: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._greedy_insertion

```{autodoc2-docstring} src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._greedy_insertion
```

````

````{py:method} _regret_2_insertion(partial_routes: typing.List[typing.List[int]], removed: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._regret_2_insertion

```{autodoc2-docstring} src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._regret_2_insertion
```

````

````{py:method} _build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._build_initial_solution

```{autodoc2-docstring} src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._build_initial_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._evaluate

```{autodoc2-docstring} src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._cost

```{autodoc2-docstring} src.models.core.dr_alns.dr_alns_solver.DRALNSSolver._cost
```

````

`````
