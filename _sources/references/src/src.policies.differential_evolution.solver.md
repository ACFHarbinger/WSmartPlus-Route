# {py:mod}`src.policies.differential_evolution.solver`

```{py:module} src.policies.differential_evolution.solver
```

```{autodoc2-docstring} src.policies.differential_evolution.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DESolver <src.policies.differential_evolution.solver.DESolver>`
  - ```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver
    :summary:
    ```
````

### API

`````{py:class} DESolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.differential_evolution.params.DEParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.differential_evolution.solver.DESolver

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.differential_evolution.solver.DESolver.solve

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver.solve
```

````

````{py:method} _generate_mutation_indices(pop_size: int) -> typing.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
:canonical: src.policies.differential_evolution.solver.DESolver._generate_mutation_indices

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver._generate_mutation_indices
```

````

````{py:method} _vectorized_exponential_crossover(target_pop: numpy.ndarray, mutant_pop: numpy.ndarray) -> numpy.ndarray
:canonical: src.policies.differential_evolution.solver.DESolver._vectorized_exponential_crossover

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver._vectorized_exponential_crossover
```

````

````{py:method} _initialize_population() -> numpy.ndarray
:canonical: src.policies.differential_evolution.solver.DESolver._initialize_population

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver._initialize_population
```

````

````{py:method} _encode_routes(routes: typing.List[typing.List[int]]) -> numpy.ndarray
:canonical: src.policies.differential_evolution.solver.DESolver._encode_routes

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver._encode_routes
```

````

````{py:method} _decode_vector(vector: numpy.ndarray) -> typing.List[typing.List[int]]
:canonical: src.policies.differential_evolution.solver.DESolver._decode_vector

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver._decode_vector
```

````

````{py:method} _apply_boundary_handling(vector: numpy.ndarray, base_vector: numpy.ndarray) -> numpy.ndarray
:canonical: src.policies.differential_evolution.solver.DESolver._apply_boundary_handling

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver._apply_boundary_handling
```

````

````{py:method} _exponential_crossover(target: numpy.ndarray, mutant: numpy.ndarray) -> numpy.ndarray
:canonical: src.policies.differential_evolution.solver.DESolver._exponential_crossover

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver._exponential_crossover
```

````

````{py:method} _differential_mutation(base: numpy.ndarray, diff1: numpy.ndarray, diff2: numpy.ndarray, F: float) -> numpy.ndarray
:canonical: src.policies.differential_evolution.solver.DESolver._differential_mutation

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver._differential_mutation
```

````

````{py:method} _binomial_crossover(target: numpy.ndarray, mutant: numpy.ndarray, CR: float) -> numpy.ndarray
:canonical: src.policies.differential_evolution.solver.DESolver._binomial_crossover

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver._binomial_crossover
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.differential_evolution.solver.DESolver._evaluate

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.differential_evolution.solver.DESolver._cost

```{autodoc2-docstring} src.policies.differential_evolution.solver.DESolver._cost
```

````

`````
