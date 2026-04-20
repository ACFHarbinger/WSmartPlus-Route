# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.subproblem`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.subproblem
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.subproblem
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RoutingSubproblem <src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.subproblem.RoutingSubproblem>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.subproblem.RoutingSubproblem
    :summary:
    ```
````

### API

`````{py:class} RoutingSubproblem(distance_matrix: numpy.ndarray, timeout_seconds: float = 10.0)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.subproblem.RoutingSubproblem

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.subproblem.RoutingSubproblem
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.subproblem.RoutingSubproblem.__init__
```

````{py:method} solve(assigned_nodes: typing.List[int]) -> typing.Tuple[bool, float, typing.List[int]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.subproblem.RoutingSubproblem.solve

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.logic_based_benders_decomposition.subproblem.RoutingSubproblem.solve
```

````

`````
