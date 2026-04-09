# {py:mod}`src.policies.branch_and_price.rcspp_dp`

```{py:module} src.policies.branch_and_price.rcspp_dp
```

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Label <src.policies.branch_and_price.rcspp_dp.Label>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.Label
    :summary:
    ```
* - {py:obj}`RCSPPSolver <src.policies.branch_and_price.rcspp_dp.RCSPPSolver>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.RCSPPSolver
    :summary:
    ```
````

### API

`````{py:class} Label
:canonical: src.policies.branch_and_price.rcspp_dp.Label

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.Label
```

````{py:attribute} reduced_cost
:canonical: src.policies.branch_and_price.rcspp_dp.Label.reduced_cost
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.Label.reduced_cost
```

````

````{py:attribute} node
:canonical: src.policies.branch_and_price.rcspp_dp.Label.node
:type: int
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.Label.node
```

````

````{py:attribute} cost
:canonical: src.policies.branch_and_price.rcspp_dp.Label.cost
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.Label.cost
```

````

````{py:attribute} load
:canonical: src.policies.branch_and_price.rcspp_dp.Label.load
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.Label.load
```

````

````{py:attribute} revenue
:canonical: src.policies.branch_and_price.rcspp_dp.Label.revenue
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.Label.revenue
```

````

````{py:attribute} path
:canonical: src.policies.branch_and_price.rcspp_dp.Label.path
:type: typing.List[int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.Label.path
```

````

````{py:attribute} visited
:canonical: src.policies.branch_and_price.rcspp_dp.Label.visited
:type: typing.Set[int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.Label.visited
```

````

````{py:attribute} ng_memory
:canonical: src.policies.branch_and_price.rcspp_dp.Label.ng_memory
:type: typing.Set[int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.Label.ng_memory
```

````

````{py:attribute} rf_unmatched
:canonical: src.policies.branch_and_price.rcspp_dp.Label.rf_unmatched
:type: typing.FrozenSet[int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.Label.rf_unmatched
```

````

````{py:attribute} parent
:canonical: src.policies.branch_and_price.rcspp_dp.Label.parent
:type: typing.Optional[src.policies.branch_and_price.rcspp_dp.Label]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.Label.parent
```

````

````{py:attribute} sri_state
:canonical: src.policies.branch_and_price.rcspp_dp.Label.sri_state
:type: typing.Tuple[int, ...]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.Label.sri_state
```

````

````{py:method} dominates(other: src.policies.branch_and_price.rcspp_dp.Label, use_ng: bool = False, epsilon: float = 1e-06) -> bool
:canonical: src.policies.branch_and_price.rcspp_dp.Label.dominates

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.Label.dominates
```

````

````{py:method} is_feasible(capacity: float) -> bool
:canonical: src.policies.branch_and_price.rcspp_dp.Label.is_feasible

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.Label.is_feasible
```

````

````{py:method} reconstruct_path() -> typing.List[int]
:canonical: src.policies.branch_and_price.rcspp_dp.Label.reconstruct_path

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.Label.reconstruct_path
```

````

`````

`````{py:class} RCSPPSolver(n_nodes: int, cost_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue_per_kg: float, cost_per_km: float, mandatory_nodes: typing.Optional[typing.Set[int]] = None, use_ng_routes: bool = True, ng_neighborhood_size: int = 8, ng_neighborhoods: typing.Optional[typing.Dict[int, typing.Set[int]]] = None)
:canonical: src.policies.branch_and_price.rcspp_dp.RCSPPSolver

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.RCSPPSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.RCSPPSolver.__init__
```

````{py:method} _compute_ng_neighborhoods() -> typing.Dict[int, typing.Set[int]]
:canonical: src.policies.branch_and_price.rcspp_dp.RCSPPSolver._compute_ng_neighborhoods

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.RCSPPSolver._compute_ng_neighborhoods
```

````

````{py:method} solve(dual_values: typing.Dict[typing.Any, typing.Any], max_routes: int = 10, branching_constraints: typing.Optional[typing.List[typing.Any]] = None, capacity_cut_duals: typing.Optional[typing.Dict[typing.FrozenSet[int], float]] = None, sri_cut_duals: typing.Optional[typing.Dict[typing.FrozenSet[int], float]] = None, lci_cut_duals: typing.Optional[typing.Dict[typing.Tuple[int, int], float]] = None, is_farkas: bool = False) -> typing.List[src.policies.branch_and_price.master_problem.Route]
:canonical: src.policies.branch_and_price.rcspp_dp.RCSPPSolver.solve

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.RCSPPSolver.solve
```

````

````{py:method} _preprocess_constraints(constraints: typing.List[typing.Any])
:canonical: src.policies.branch_and_price.rcspp_dp.RCSPPSolver._preprocess_constraints

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.RCSPPSolver._preprocess_constraints
```

````

````{py:method} _label_correcting_algorithm(max_routes: int, forbidden_arcs: typing.FrozenSet[typing.Tuple[int, int]], required_successors: typing.Dict[int, int], required_predecessors: typing.Dict[int, int], rf_separate: typing.Set[typing.Tuple[int, int]], rf_together: typing.Set[typing.Tuple[int, int]], rcc_duals: typing.Optional[typing.Dict[typing.FrozenSet[int], float]] = None, active_sri_subsets: typing.Optional[typing.List[typing.FrozenSet[int]]] = None, sri_dual_values: typing.Optional[typing.List[float]] = None, node_to_sri: typing.Optional[typing.Dict[int, typing.List[int]]] = None) -> typing.List[src.policies.branch_and_price.master_problem.Route]
:canonical: src.policies.branch_and_price.rcspp_dp.RCSPPSolver._label_correcting_algorithm

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.RCSPPSolver._label_correcting_algorithm
```

````

````{py:method} _extend_label(label: src.policies.branch_and_price.rcspp_dp.Label, next_node: int, rf_together: typing.Set[typing.Tuple[int, int]], active_sri_subsets: typing.Optional[typing.List[typing.FrozenSet[int]]] = None, sri_dual_values: typing.Optional[typing.List[float]] = None, node_to_sri: typing.Optional[typing.Dict[int, typing.List[int]]] = None) -> typing.Optional[src.policies.branch_and_price.rcspp_dp.Label]
:canonical: src.policies.branch_and_price.rcspp_dp.RCSPPSolver._extend_label

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.RCSPPSolver._extend_label
```

````

````{py:method} _extend_to_depot(label: src.policies.branch_and_price.rcspp_dp.Label) -> typing.Optional[src.policies.branch_and_price.rcspp_dp.Label]
:canonical: src.policies.branch_and_price.rcspp_dp.RCSPPSolver._extend_to_depot

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.RCSPPSolver._extend_to_depot
```

````

````{py:method} _is_dominated(label: src.policies.branch_and_price.rcspp_dp.Label, existing: typing.List[src.policies.branch_and_price.rcspp_dp.Label], use_ng: bool) -> bool
:canonical: src.policies.branch_and_price.rcspp_dp.RCSPPSolver._is_dominated

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.RCSPPSolver._is_dominated
```

````

````{py:method} compute_route_details(route: typing.List[int])
:canonical: src.policies.branch_and_price.rcspp_dp.RCSPPSolver.compute_route_details

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.RCSPPSolver.compute_route_details
```

````

````{py:method} get_statistics()
:canonical: src.policies.branch_and_price.rcspp_dp.RCSPPSolver.get_statistics

```{autodoc2-docstring} src.policies.branch_and_price.rcspp_dp.RCSPPSolver.get_statistics
```

````

`````
