# {py:mod}`src.policies.league_championship_algorithm.solver`

```{py:module} src.policies.league_championship_algorithm.solver
```

```{autodoc2-docstring} src.policies.league_championship_algorithm.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LCASolver <src.policies.league_championship_algorithm.solver.LCASolver>`
  - ```{autodoc2-docstring} src.policies.league_championship_algorithm.solver.LCASolver
    :summary:
    ```
````

### API

`````{py:class} LCASolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.league_championship_algorithm.params.LCAParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.league_championship_algorithm.solver.LCASolver

```{autodoc2-docstring} src.policies.league_championship_algorithm.solver.LCASolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.league_championship_algorithm.solver.LCASolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.league_championship_algorithm.solver.LCASolver.solve

```{autodoc2-docstring} src.policies.league_championship_algorithm.solver.LCASolver.solve
```

````

````{py:method} _build_random_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.league_championship_algorithm.solver.LCASolver._build_random_solution

```{autodoc2-docstring} src.policies.league_championship_algorithm.solver.LCASolver._build_random_solution
```

````

````{py:method} _perturb(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.league_championship_algorithm.solver.LCASolver._perturb

```{autodoc2-docstring} src.policies.league_championship_algorithm.solver.LCASolver._perturb
```

````

````{py:method} _crossover(loser_routes: typing.List[typing.List[int]], winner_routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.league_championship_algorithm.solver.LCASolver._crossover

```{autodoc2-docstring} src.policies.league_championship_algorithm.solver.LCASolver._crossover
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.league_championship_algorithm.solver.LCASolver._evaluate

```{autodoc2-docstring} src.policies.league_championship_algorithm.solver.LCASolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.league_championship_algorithm.solver.LCASolver._cost

```{autodoc2-docstring} src.policies.league_championship_algorithm.solver.LCASolver._cost
```

````

`````
