# {py:mod}`src.policies.route_construction.meta_heuristics.genius.solver`

```{py:module} src.policies.route_construction.meta_heuristics.genius.solver
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genius.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GENIUSSolver <src.policies.route_construction.meta_heuristics.genius.solver.GENIUSSolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genius.solver.GENIUSSolver
    :summary:
    ```
````

### API

`````{py:class} GENIUSSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.meta_heuristics.genius.params.GENIUSParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.meta_heuristics.genius.solver.GENIUSSolver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genius.solver.GENIUSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genius.solver.GENIUSSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.genius.solver.GENIUSSolver.solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genius.solver.GENIUSSolver.solve
```

````

````{py:method} _build_initial_solution_geni() -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.genius.solver.GENIUSSolver._build_initial_solution_geni

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genius.solver.GENIUSSolver._build_initial_solution_geni
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.genius.solver.GENIUSSolver._evaluate

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genius.solver.GENIUSSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.genius.solver.GENIUSSolver._cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genius.solver.GENIUSSolver._cost
```

````

`````
