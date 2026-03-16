# {py:mod}`src.policies.memetic_algorithm.solver`

```{py:module} src.policies.memetic_algorithm.solver
```

```{autodoc2-docstring} src.policies.memetic_algorithm.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MASolver <src.policies.memetic_algorithm.solver.MASolver>`
  - ```{autodoc2-docstring} src.policies.memetic_algorithm.solver.MASolver
    :summary:
    ```
````

### API

`````{py:class} MASolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.memetic_algorithm.params.MAParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.memetic_algorithm.solver.MASolver

```{autodoc2-docstring} src.policies.memetic_algorithm.solver.MASolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.memetic_algorithm.solver.MASolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.memetic_algorithm.solver.MASolver.solve

```{autodoc2-docstring} src.policies.memetic_algorithm.solver.MASolver.solve
```

````

````{py:method} _select_from_population(population: typing.List[typing.List[typing.List[int]]]) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.memetic_algorithm.solver.MASolver._select_from_population

```{autodoc2-docstring} src.policies.memetic_algorithm.solver.MASolver._select_from_population
```

````

````{py:method} _generate_new_population(breeders: typing.List[typing.List[typing.List[int]]]) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.memetic_algorithm.solver.MASolver._generate_new_population

```{autodoc2-docstring} src.policies.memetic_algorithm.solver.MASolver._generate_new_population
```

````

````{py:method} _update_population(old_pop: typing.List[typing.List[typing.List[int]]], new_pop: typing.List[typing.List[typing.List[int]]]) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.memetic_algorithm.solver.MASolver._update_population

```{autodoc2-docstring} src.policies.memetic_algorithm.solver.MASolver._update_population
```

````

````{py:method} _local_improver(solution: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.memetic_algorithm.solver.MASolver._local_improver

```{autodoc2-docstring} src.policies.memetic_algorithm.solver.MASolver._local_improver
```

````

````{py:method} _recombination(parent1: typing.List[typing.List[int]], parent2: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.memetic_algorithm.solver.MASolver._recombination

```{autodoc2-docstring} src.policies.memetic_algorithm.solver.MASolver._recombination
```

````

````{py:method} _mutation(solution: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.memetic_algorithm.solver.MASolver._mutation

```{autodoc2-docstring} src.policies.memetic_algorithm.solver.MASolver._mutation
```

````

````{py:method} _initialize_population() -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.memetic_algorithm.solver.MASolver._initialize_population

```{autodoc2-docstring} src.policies.memetic_algorithm.solver.MASolver._initialize_population
```

````

````{py:method} _evaluate(solution: typing.List[typing.List[int]]) -> float
:canonical: src.policies.memetic_algorithm.solver.MASolver._evaluate

```{autodoc2-docstring} src.policies.memetic_algorithm.solver.MASolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.memetic_algorithm.solver.MASolver._cost

```{autodoc2-docstring} src.policies.memetic_algorithm.solver.MASolver._cost
```

````

`````
