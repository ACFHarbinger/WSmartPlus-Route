# {py:mod}`src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver`

```{py:module} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PSOMASolver <src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver
    :summary:
    ```
````

### API

`````{py:class} PSOMASolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.params.PSOMAParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver.solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver.solve
```

````

````{py:method} _init_swarm()
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._init_swarm

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._init_swarm
```

````

````{py:method} _set_gbest(X: numpy.ndarray, giant_tour: numpy.ndarray, mapping: numpy.ndarray, routes: typing.List[typing.List[int]], profit: float)
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._set_gbest

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._set_gbest
```

````

````{py:method} _calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._calculate_cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._calculate_cost
```

````

````{py:method} _training_phase() -> None
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._training_phase

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._training_phase
```

````

````{py:method} _non_training_phase() -> None
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._non_training_phase

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._non_training_phase
```

````

````{py:method} _sa_search(operator: typing.Callable) -> typing.Tuple[float, numpy.ndarray, numpy.ndarray, typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._sa_search

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._sa_search
```

````

````{py:method} _update_probabilities() -> None
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._update_probabilities

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._update_probabilities
```

````

````{py:method} _swap(tour: numpy.ndarray, X: numpy.ndarray, mapping: numpy.ndarray) -> typing.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._swap

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._swap
```

````

````{py:method} _insert(tour: numpy.ndarray, X: numpy.ndarray, mapping: numpy.ndarray) -> typing.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._insert

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._insert
```

````

````{py:method} _inverse(tour: numpy.ndarray, X: numpy.ndarray, mapping: numpy.ndarray) -> typing.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._inverse

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver._inverse
```

````

`````
