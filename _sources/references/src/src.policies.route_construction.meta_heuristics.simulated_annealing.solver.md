# {py:mod}`src.policies.route_construction.meta_heuristics.simulated_annealing.solver`

```{py:module} src.policies.route_construction.meta_heuristics.simulated_annealing.solver
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SASolver <src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver
    :summary:
    ```
````

### API

`````{py:class} SASolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.meta_heuristics.simulated_annealing.params.SAParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver

Bases: {py:obj}`logic.src.policies.helpers.local_search.local_search_base.LocalSearch`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver.__init__
```

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver._evaluate

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver._cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver._cost
```

````

````{py:method} _get_route_arcs(route: typing.List[int]) -> typing.List[typing.Tuple[int, int]]
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver._get_route_arcs

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver._get_route_arcs
```

````

````{py:method} _get_affected_arcs(op: str, current_routes: typing.List[typing.List[int]], u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> typing.Tuple[typing.List[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int]]]
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver._get_affected_arcs

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver._get_affected_arcs
```

````

````{py:method} _delta_evaluate(removed: typing.List[typing.Tuple[int, int]], inserted: typing.List[typing.Tuple[int, int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver._delta_evaluate

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver._delta_evaluate
```

````

````{py:method} _perturb(current_routes: typing.List[typing.List[int]], T: float, T_0: float) -> typing.Tuple[str, typing.List[typing.List[int]], float]
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver._perturb

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver._perturb
```

````

````{py:method} _calibrate_initial_temperature(start_routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver._calibrate_initial_temperature

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver._calibrate_initial_temperature
```

````

````{py:method} optimize(solution: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver.optimize

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver.optimize
```

````

````{py:method} solve(initial_solution: typing.Optional[typing.List[typing.List[int]]] = None) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver.solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing.solver.SASolver.solve
```

````

`````
