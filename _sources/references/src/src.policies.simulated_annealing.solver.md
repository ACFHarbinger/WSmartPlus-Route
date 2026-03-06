# {py:mod}`src.policies.simulated_annealing.solver`

```{py:module} src.policies.simulated_annealing.solver
```

```{autodoc2-docstring} src.policies.simulated_annealing.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SASolver <src.policies.simulated_annealing.solver.SASolver>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing.solver.SASolver
    :summary:
    ```
````

### API

`````{py:class} SASolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.simulated_annealing.params.SAParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.simulated_annealing.solver.SASolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.simulated_annealing.solver.SASolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.simulated_annealing.solver.SASolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.simulated_annealing.solver.SASolver.solve

```{autodoc2-docstring} src.policies.simulated_annealing.solver.SASolver.solve
```

````

````{py:method} _llh0(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.simulated_annealing.solver.SASolver._llh0

```{autodoc2-docstring} src.policies.simulated_annealing.solver.SASolver._llh0
```

````

````{py:method} _llh1(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.simulated_annealing.solver.SASolver._llh1

```{autodoc2-docstring} src.policies.simulated_annealing.solver.SASolver._llh1
```

````

````{py:method} _llh2(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.simulated_annealing.solver.SASolver._llh2

```{autodoc2-docstring} src.policies.simulated_annealing.solver.SASolver._llh2
```

````

````{py:method} _llh3(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.simulated_annealing.solver.SASolver._llh3

```{autodoc2-docstring} src.policies.simulated_annealing.solver.SASolver._llh3
```

````

````{py:method} _llh4(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.simulated_annealing.solver.SASolver._llh4

```{autodoc2-docstring} src.policies.simulated_annealing.solver.SASolver._llh4
```

````

````{py:method} _build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.simulated_annealing.solver.SASolver._build_initial_solution

```{autodoc2-docstring} src.policies.simulated_annealing.solver.SASolver._build_initial_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.simulated_annealing.solver.SASolver._evaluate

```{autodoc2-docstring} src.policies.simulated_annealing.solver.SASolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.simulated_annealing.solver.SASolver._cost

```{autodoc2-docstring} src.policies.simulated_annealing.solver.SASolver._cost
```

````

`````
