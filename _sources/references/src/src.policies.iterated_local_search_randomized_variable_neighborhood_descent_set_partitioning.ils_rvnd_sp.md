# {py:mod}`src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp`

```{py:module} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp
```

```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ILSRVNDSPSolver <src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver>`
  - ```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.logger>`
  - ```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.logger
```

````

`````{py:class} ILSRVNDSPSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: logic.src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.params.ILSRVNDSPParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver.__init__
```

````{py:method} calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver.calculate_cost

```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver.calculate_cost
```

````

````{py:method} calculate_profit(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver.calculate_profit

```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver.calculate_profit
```

````

````{py:method} _add_to_pool(routes: typing.List[typing.List[int]], pool: typing.Optional[typing.Set[typing.Tuple[int, ...]]] = None)
:canonical: src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver._add_to_pool

```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver._add_to_pool
```

````

````{py:method} _perturb(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver._perturb

```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver._perturb
```

````

````{py:method} build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver.build_initial_solution

```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver.build_initial_solution
```

````

````{py:method} solve_set_partitioning(pool: typing.Optional[typing.Set[typing.Tuple[int, ...]]] = None) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver.solve_set_partitioning

```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver.solve_set_partitioning
```

````

````{py:method} run_ils_rvnd(initial_routes: typing.List[typing.List[int]], max_iterations: int, max_ils_iterations: int, target_pool: typing.Set[typing.Tuple[int, ...]], tolerance: float, start_time: float, rvnd: logic.src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.rvnd.RVND) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver.run_ils_rvnd

```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver.run_ils_rvnd
```

````

````{py:method} _run_strategy_a(initial_solution: typing.Optional[typing.List[typing.List[int]]], start_time: float, rvnd: logic.src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.rvnd.RVND) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver._run_strategy_a

```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver._run_strategy_a
```

````

````{py:method} _run_strategy_b(initial_solution: typing.Optional[typing.List[typing.List[int]]], start_time: float, rvnd: logic.src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.rvnd.RVND) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver._run_strategy_b

```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver._run_strategy_b
```

````

````{py:method} solve(initial_solution: typing.Optional[typing.List[typing.List[int]]] = None) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver.solve

```{autodoc2-docstring} src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.ils_rvnd_sp.ILSRVNDSPSolver.solve
```

````

`````
