# {py:mod}`src.policies.late_acceptance_hill_climbing.solver`

```{py:module} src.policies.late_acceptance_hill_climbing.solver
```

```{autodoc2-docstring} src.policies.late_acceptance_hill_climbing.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LAHCSolver <src.policies.late_acceptance_hill_climbing.solver.LAHCSolver>`
  - ```{autodoc2-docstring} src.policies.late_acceptance_hill_climbing.solver.LAHCSolver
    :summary:
    ```
````

### API

`````{py:class} LAHCSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.late_acceptance_hill_climbing.params.LAHCParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.late_acceptance_hill_climbing.solver.LAHCSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.late_acceptance_hill_climbing.solver.LAHCSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.late_acceptance_hill_climbing.solver.LAHCSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.late_acceptance_hill_climbing.solver.LAHCSolver.solve

```{autodoc2-docstring} src.policies.late_acceptance_hill_climbing.solver.LAHCSolver.solve
```

````

````{py:method} _llh0(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.late_acceptance_hill_climbing.solver.LAHCSolver._llh0

```{autodoc2-docstring} src.policies.late_acceptance_hill_climbing.solver.LAHCSolver._llh0
```

````

````{py:method} _llh1(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.late_acceptance_hill_climbing.solver.LAHCSolver._llh1

```{autodoc2-docstring} src.policies.late_acceptance_hill_climbing.solver.LAHCSolver._llh1
```

````

````{py:method} _llh2(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.late_acceptance_hill_climbing.solver.LAHCSolver._llh2

```{autodoc2-docstring} src.policies.late_acceptance_hill_climbing.solver.LAHCSolver._llh2
```

````

````{py:method} _llh3(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.late_acceptance_hill_climbing.solver.LAHCSolver._llh3

```{autodoc2-docstring} src.policies.late_acceptance_hill_climbing.solver.LAHCSolver._llh3
```

````

````{py:method} _llh4(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.late_acceptance_hill_climbing.solver.LAHCSolver._llh4

```{autodoc2-docstring} src.policies.late_acceptance_hill_climbing.solver.LAHCSolver._llh4
```

````

````{py:method} _build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.late_acceptance_hill_climbing.solver.LAHCSolver._build_initial_solution

```{autodoc2-docstring} src.policies.late_acceptance_hill_climbing.solver.LAHCSolver._build_initial_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.late_acceptance_hill_climbing.solver.LAHCSolver._evaluate

```{autodoc2-docstring} src.policies.late_acceptance_hill_climbing.solver.LAHCSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.late_acceptance_hill_climbing.solver.LAHCSolver._cost

```{autodoc2-docstring} src.policies.late_acceptance_hill_climbing.solver.LAHCSolver._cost
```

````

`````
