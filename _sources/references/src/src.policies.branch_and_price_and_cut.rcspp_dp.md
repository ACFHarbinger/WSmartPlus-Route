# {py:mod}`src.policies.branch_and_price_and_cut.rcspp_dp`

```{py:module} src.policies.branch_and_price_and_cut.rcspp_dp
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Label <src.policies.branch_and_price_and_cut.rcspp_dp.Label>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.Label
    :summary:
    ```
* - {py:obj}`RCSPPSolver <src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver
    :summary:
    ```
````

### API

`````{py:class} Label
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.Label

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.Label
```

````{py:attribute} reduced_cost
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.Label.reduced_cost
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.Label.reduced_cost
```

````

````{py:attribute} node
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.Label.node
:type: int
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.Label.node
```

````

````{py:attribute} load
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.Label.load
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.Label.load
```

````

````{py:attribute} visited
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.Label.visited
:type: typing.Set[int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.Label.visited
```

````

````{py:attribute} ng_memory
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.Label.ng_memory
:type: typing.Set[int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.Label.ng_memory
```

````

````{py:attribute} rf_unmatched
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.Label.rf_unmatched
:type: typing.FrozenSet[int]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.Label.rf_unmatched
```

````

````{py:attribute} parent
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.Label.parent
:type: typing.Optional[src.policies.branch_and_price_and_cut.rcspp_dp.Label]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.Label.parent
```

````

````{py:attribute} sri_state
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.Label.sri_state
:type: typing.Tuple[int, ...]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.Label.sri_state
```

````

````{py:method} dominates(other: src.policies.branch_and_price_and_cut.rcspp_dp.Label, use_ng: bool = False, epsilon: float = 1e-06, sri_dual_values: typing.Optional[typing.List[float]] = None) -> bool
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.Label.dominates

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.Label.dominates
```

````

````{py:method} is_feasible(capacity: float) -> bool
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.Label.is_feasible

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.Label.is_feasible
```

````

````{py:method} reconstruct_path() -> typing.List[int]
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.Label.reconstruct_path

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.Label.reconstruct_path
```

````

`````

`````{py:class} RCSPPSolver(n_nodes: int, cost_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue_per_kg: float, cost_per_km: float, mandatory_nodes: typing.Optional[typing.Set[int]] = None, use_ng_routes: bool = True, ng_neighborhood_size: int = 8, ng_neighborhoods: typing.Optional[typing.Dict[int, typing.Set[int]]] = None)
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver.__init__
```

````{py:method} _precompute_sorted_neighbors() -> typing.Dict[int, typing.List[int]]
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver._precompute_sorted_neighbors

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver._precompute_sorted_neighbors
```

````

````{py:method} _compute_ng_neighborhoods() -> typing.Dict[int, typing.Set[int]]
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver._compute_ng_neighborhoods

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver._compute_ng_neighborhoods
```

````

````{py:method} expand_ng_neighborhoods(cycles: typing.List[typing.Tuple[int, ...]]) -> int
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver.expand_ng_neighborhoods

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver.expand_ng_neighborhoods
```

````

````{py:method} enforce_elementarity(nodes: typing.List[int]) -> int
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver.enforce_elementarity

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver.enforce_elementarity
```

````

````{py:method} solve(dual_values: typing.Union[typing.Dict[int, float], typing.Dict[str, typing.Any]], max_routes: int = 10, branching_constraints: typing.Optional[typing.List[src.policies.branch_and_price_and_cut.branching.AnyBranchingConstraint]] = None, capacity_cut_duals: typing.Optional[typing.Dict[typing.FrozenSet[int], float]] = None, sri_cut_duals: typing.Optional[typing.Dict[typing.FrozenSet[int], float]] = None, edge_clique_cut_duals: typing.Optional[typing.Dict[typing.Tuple[int, int], float]] = None, forced_nodes: typing.Optional[typing.Set[int]] = None, rf_conflicts: typing.Optional[typing.Dict[int, typing.Set[int]]] = None, is_farkas: bool = False, max_active_sris: int = 15) -> typing.List[src.policies.branch_and_price_and_cut.master_problem.Route]
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver.solve

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver.solve
```

````

````{py:method} _compute_completion_bounds()
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver._compute_completion_bounds

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver._compute_completion_bounds
```

````

````{py:method} compute_route_details(route_nodes: typing.List[int]) -> src.policies.branch_and_price_and_cut.master_problem.Route
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver.compute_route_details

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver.compute_route_details
```

````

````{py:method} _preprocess_constraints(constraints: typing.List[typing.Any])
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver._preprocess_constraints

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver._preprocess_constraints
```

````

````{py:method} _label_correcting_algorithm(max_routes: int, forbidden_arcs: typing.FrozenSet[typing.Tuple[int, int]], required_successors: typing.Dict[int, int], required_predecessors: typing.Dict[int, int], rf_separate: typing.Set[typing.Tuple[int, int]], rf_together: typing.Set[typing.Tuple[int, int]], rcc_duals: typing.Dict[typing.FrozenSet[int], float], active_sri_subsets: typing.List[typing.FrozenSet[int]], sri_dual_values: typing.List[float], node_to_sri: typing.Dict[int, typing.List[int]], forced_nodes: typing.Set[int], sri_memory_nodes: typing.Optional[typing.List[typing.Set[int]]] = None) -> typing.List[src.policies.branch_and_price_and_cut.master_problem.Route]
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver._label_correcting_algorithm

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver._label_correcting_algorithm
```

````

````{py:method} _get_neighbors(node: int, limit: int) -> typing.List[int]
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver._get_neighbors

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver._get_neighbors
```

````

````{py:method} _extend_to_depot(label: src.policies.branch_and_price_and_cut.rcspp_dp.Label) -> typing.Optional[src.policies.branch_and_price_and_cut.rcspp_dp.Label]
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver._extend_to_depot

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver._extend_to_depot
```

````

````{py:method} _route_details_from_path(route_nodes: typing.List[int])
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver._route_details_from_path

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver._route_details_from_path
```

````

````{py:method} _extend_label(label: src.policies.branch_and_price_and_cut.rcspp_dp.Label, next_node: int, forbidden: typing.FrozenSet[typing.Tuple[int, int]], rcc_duals: typing.Dict[typing.FrozenSet[int], float], active_sri: typing.List[typing.FrozenSet[int]], sri_duals: typing.List[float], node_to_sri: typing.Dict[int, typing.List[int]], sri_memory_nodes: typing.Optional[typing.List[typing.Set[int]]] = None) -> typing.Optional[src.policies.branch_and_price_and_cut.rcspp_dp.Label]
:canonical: src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver._extend_label

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver._extend_label
```

````

`````
