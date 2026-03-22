# {py:mod}`src.policies.particle_swarm_optimization_memetic_algorithm.solver`

```{py:module} src.policies.particle_swarm_optimization_memetic_algorithm.solver
```

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic_algorithm.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PSOMAsSolver <src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver>`
  - ```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver
    :summary:
    ```
````

### API

`````{py:class} PSOMAsSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.particle_swarm_optimization_memetic_algorithm.params.PSOMAParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver.solve

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver.solve
```

````

````{py:method} _init_swarm() -> typing.List[src.policies.particle_swarm_optimization_memetic_algorithm.particle.PSOMAParticle]
:canonical: src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._init_swarm

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._init_swarm
```

````

````{py:method} _build_random_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._build_random_solution

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._build_random_solution
```

````

````{py:method} _global_best(swarm: typing.List[src.policies.particle_swarm_optimization_memetic_algorithm.particle.PSOMAParticle]) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._global_best

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._global_best
```

````

````{py:method} _update_position(current: typing.List[typing.List[int]], pbest: typing.List[typing.List[int]], gbest: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._update_position

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._update_position
```

````

````{py:method} _apply_velocity(current: typing.List[typing.List[int]], target: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._apply_velocity

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._apply_velocity
```

````

````{py:method} _partition_flat(flat_nodes: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._partition_flat

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._partition_flat
```

````

````{py:method} _random_relocate(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._random_relocate

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._random_relocate
```

````

````{py:method} _local_search(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._local_search

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._local_search
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._evaluate

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._cost

```{autodoc2-docstring} src.policies.particle_swarm_optimization_memetic_algorithm.solver.PSOMAsSolver._cost
```

````

`````
