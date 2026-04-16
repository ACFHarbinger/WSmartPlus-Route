# {py:mod}`src.policies.route_construction.meta_heuristics.genetic_algorithm.solver`

```{py:module} src.policies.route_construction.meta_heuristics.genetic_algorithm.solver
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GASolver <src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver
    :summary:
    ```
````

### API

`````{py:class} GASolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.meta_heuristics.genetic_algorithm.params.GAParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver.solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver.solve
```

````

````{py:method} _init_population() -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver._init_population

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver._init_population
```

````

````{py:method} _tournament_select(population: typing.List[typing.List[typing.List[int]]], fitnesses: typing.List[float]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver._tournament_select

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver._tournament_select
```

````

````{py:method} _crossover(parent1: typing.List[typing.List[int]], parent2: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver._crossover

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver._crossover
```

````

````{py:method} _mutate(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver._mutate

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver._mutate
```

````

````{py:method} _local_search(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver._local_search

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver._local_search
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver._evaluate

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver._cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.solver.GASolver._cost
```

````

`````
