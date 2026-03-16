# {py:mod}`src.policies.artificial_bee_colony.solver`

```{py:module} src.policies.artificial_bee_colony.solver
```

```{autodoc2-docstring} src.policies.artificial_bee_colony.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ABCSolver <src.policies.artificial_bee_colony.solver.ABCSolver>`
  - ```{autodoc2-docstring} src.policies.artificial_bee_colony.solver.ABCSolver
    :summary:
    ```
````

### API

`````{py:class} ABCSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.artificial_bee_colony.params.ABCParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.artificial_bee_colony.solver.ABCSolver

```{autodoc2-docstring} src.policies.artificial_bee_colony.solver.ABCSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.artificial_bee_colony.solver.ABCSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.artificial_bee_colony.solver.ABCSolver.solve

```{autodoc2-docstring} src.policies.artificial_bee_colony.solver.ABCSolver.solve
```

````

````{py:method} _new_source() -> typing.List[typing.List[int]]
:canonical: src.policies.artificial_bee_colony.solver.ABCSolver._new_source

```{autodoc2-docstring} src.policies.artificial_bee_colony.solver.ABCSolver._new_source
```

````

````{py:method} _build_random_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.artificial_bee_colony.solver.ABCSolver._build_random_solution

```{autodoc2-docstring} src.policies.artificial_bee_colony.solver.ABCSolver._build_random_solution
```

````

````{py:method} _perturb(current: typing.List[typing.List[int]], peer: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.artificial_bee_colony.solver.ABCSolver._perturb

```{autodoc2-docstring} src.policies.artificial_bee_colony.solver.ABCSolver._perturb
```

````

````{py:method} _roulette(probs: typing.List[float], rng: random.Random) -> int
:canonical: src.policies.artificial_bee_colony.solver.ABCSolver._roulette
:staticmethod:

```{autodoc2-docstring} src.policies.artificial_bee_colony.solver.ABCSolver._roulette
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.artificial_bee_colony.solver.ABCSolver._evaluate

```{autodoc2-docstring} src.policies.artificial_bee_colony.solver.ABCSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.artificial_bee_colony.solver.ABCSolver._cost

```{autodoc2-docstring} src.policies.artificial_bee_colony.solver.ABCSolver._cost
```

````

`````
