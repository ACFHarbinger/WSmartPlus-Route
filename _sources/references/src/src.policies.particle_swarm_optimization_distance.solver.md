# {py:mod}`src.policies.particle_swarm_optimization_distance.solver`

```{py:module} src.policies.particle_swarm_optimization_distance.solver
```

```{autodoc2-docstring} src.policies.particle_swarm_optimization_distance.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DistancePSOSolver <src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver>`
  - ```{autodoc2-docstring} src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver
    :summary:
    ```
````

### API

`````{py:class} DistancePSOSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.particle_swarm_optimization_distance.params.DistancePSOParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver.solve

```{autodoc2-docstring} src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver.solve
```

````

````{py:method} _initialize_particle() -> typing.List[typing.List[int]]
:canonical: src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver._initialize_particle

```{autodoc2-docstring} src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver._initialize_particle
```

````

````{py:method} _update_velocity(current_position: typing.List[typing.List[int]], current_velocity: typing.Set[int], personal_best: typing.List[typing.List[int]], global_best: typing.List[typing.List[int]], inertia_weight: float) -> typing.Set[int]
:canonical: src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver._update_velocity

```{autodoc2-docstring} src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver._update_velocity
```

````

````{py:method} _apply_velocity(current_position: typing.List[typing.List[int]], velocity: typing.Set[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver._apply_velocity

```{autodoc2-docstring} src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver._apply_velocity
```

````

````{py:method} _get_node_set(routes: typing.List[typing.List[int]]) -> typing.Set[int]
:canonical: src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver._get_node_set
:staticmethod:

```{autodoc2-docstring} src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver._get_node_set
```

````

````{py:method} _compute_best_insertion_cost(node: int, routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver._compute_best_insertion_cost

```{autodoc2-docstring} src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver._compute_best_insertion_cost
```

````

````{py:method} _random_walk(particle: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver._random_walk

```{autodoc2-docstring} src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver._random_walk
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver._evaluate

```{autodoc2-docstring} src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver._cost

```{autodoc2-docstring} src.policies.particle_swarm_optimization_distance.solver.DistancePSOSolver._cost
```

````

`````
