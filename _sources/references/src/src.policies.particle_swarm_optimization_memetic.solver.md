# {py:mod}`src.policies.particle_swarm_optimization_memetic.solver`

```{py:module} src.policies.particle_swarm_optimization_memetic.solver
```

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PSOMAsSolver <src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver>`
  - ```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver
    :summary:
    ```
````

### API

`````{py:class} PSOMAsSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.particle_swarm_optimization_memetic.params.PSOMAParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver.solve

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver.solve
```

````

````{py:method} _init_swarm() -> typing.List[src.policies.particle_swarm_optimization_memetic.particle.PSOMAParticle]
:canonical: src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver._init_swarm

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver._init_swarm
```

````

````{py:method} _build_random_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver._build_random_solution

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver._build_random_solution
```

````

````{py:method} _global_best(swarm: typing.List[src.policies.particle_swarm_optimization_memetic.particle.PSOMAParticle]) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver._global_best

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver._global_best
```

````

````{py:method} _update_position(current: typing.List[typing.List[int]], pbest: typing.List[typing.List[int]], gbest: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver._update_position

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver._update_position
```

````

````{py:method} _crossover(base_routes: typing.List[typing.List[int]], guide_routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver._crossover

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver._crossover
```

````

````{py:method} _random_relocate(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver._random_relocate

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver._random_relocate
```

````

````{py:method} _local_search(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver._local_search

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver._local_search
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver._evaluate

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver._cost

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic.solver.PSOMAsSolver._cost
```

````

`````
