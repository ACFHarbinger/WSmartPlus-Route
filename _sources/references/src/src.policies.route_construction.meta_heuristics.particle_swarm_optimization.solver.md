# {py:mod}`src.policies.route_construction.meta_heuristics.particle_swarm_optimization.solver`

```{py:module} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.solver
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PSOSolver <src.policies.route_construction.meta_heuristics.particle_swarm_optimization.solver.PSOSolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.solver.PSOSolver
    :summary:
    ```
````

### API

`````{py:class} PSOSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.params.PSOParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.solver.PSOSolver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.solver.PSOSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.solver.PSOSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.solver.PSOSolver.solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.solver.PSOSolver.solve
```

````

````{py:method} _decode(x: numpy.ndarray) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.solver.PSOSolver._decode

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.solver.PSOSolver._decode
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.solver.PSOSolver._evaluate

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.solver.PSOSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.particle_swarm_optimization.solver.PSOSolver._cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.particle_swarm_optimization.solver.PSOSolver._cost
```

````

`````
