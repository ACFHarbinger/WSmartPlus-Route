# {py:mod}`src.policies.variable_neighborhood_search.solver`

```{py:module} src.policies.variable_neighborhood_search.solver
```

```{autodoc2-docstring} src.policies.variable_neighborhood_search.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VNSSolver <src.policies.variable_neighborhood_search.solver.VNSSolver>`
  - ```{autodoc2-docstring} src.policies.variable_neighborhood_search.solver.VNSSolver
    :summary:
    ```
````

### API

`````{py:class} VNSSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.variable_neighborhood_search.params.VNSParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.variable_neighborhood_search.solver.VNSSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.variable_neighborhood_search.solver.VNSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.variable_neighborhood_search.solver.VNSSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.variable_neighborhood_search.solver.VNSSolver.solve

```{autodoc2-docstring} src.policies.variable_neighborhood_search.solver.VNSSolver.solve
```

````

````{py:method} _shake_n1(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.variable_neighborhood_search.solver.VNSSolver._shake_n1

```{autodoc2-docstring} src.policies.variable_neighborhood_search.solver.VNSSolver._shake_n1
```

````

````{py:method} _shake_n2(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.variable_neighborhood_search.solver.VNSSolver._shake_n2

```{autodoc2-docstring} src.policies.variable_neighborhood_search.solver.VNSSolver._shake_n2
```

````

````{py:method} _shake_n3(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.variable_neighborhood_search.solver.VNSSolver._shake_n3

```{autodoc2-docstring} src.policies.variable_neighborhood_search.solver.VNSSolver._shake_n3
```

````

````{py:method} _shake_n4(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.variable_neighborhood_search.solver.VNSSolver._shake_n4

```{autodoc2-docstring} src.policies.variable_neighborhood_search.solver.VNSSolver._shake_n4
```

````

````{py:method} _shake_n5(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.variable_neighborhood_search.solver.VNSSolver._shake_n5

```{autodoc2-docstring} src.policies.variable_neighborhood_search.solver.VNSSolver._shake_n5
```

````

````{py:method} _local_search(routes: typing.List[typing.List[int]], start: float) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.variable_neighborhood_search.solver.VNSSolver._local_search

```{autodoc2-docstring} src.policies.variable_neighborhood_search.solver.VNSSolver._local_search
```

````

````{py:method} _llh0(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.variable_neighborhood_search.solver.VNSSolver._llh0

```{autodoc2-docstring} src.policies.variable_neighborhood_search.solver.VNSSolver._llh0
```

````

````{py:method} _llh1(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.variable_neighborhood_search.solver.VNSSolver._llh1

```{autodoc2-docstring} src.policies.variable_neighborhood_search.solver.VNSSolver._llh1
```

````

````{py:method} _llh2(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.variable_neighborhood_search.solver.VNSSolver._llh2

```{autodoc2-docstring} src.policies.variable_neighborhood_search.solver.VNSSolver._llh2
```

````

````{py:method} _llh3(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.variable_neighborhood_search.solver.VNSSolver._llh3

```{autodoc2-docstring} src.policies.variable_neighborhood_search.solver.VNSSolver._llh3
```

````

````{py:method} _llh4(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.variable_neighborhood_search.solver.VNSSolver._llh4

```{autodoc2-docstring} src.policies.variable_neighborhood_search.solver.VNSSolver._llh4
```

````

````{py:method} _build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.variable_neighborhood_search.solver.VNSSolver._build_initial_solution

```{autodoc2-docstring} src.policies.variable_neighborhood_search.solver.VNSSolver._build_initial_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.variable_neighborhood_search.solver.VNSSolver._evaluate

```{autodoc2-docstring} src.policies.variable_neighborhood_search.solver.VNSSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.variable_neighborhood_search.solver.VNSSolver._cost

```{autodoc2-docstring} src.policies.variable_neighborhood_search.solver.VNSSolver._cost
```

````

`````
