# {py:mod}`src.policies.helpers.branching_solvers.pricing.solver`

```{py:module} src.policies.helpers.branching_solvers.pricing.solver
```

```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RCSPPSolver <src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver>`
  - ```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.helpers.branching_solvers.pricing.solver.logger>`
  - ```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver.logger
    :summary:
    ```
* - {py:obj}`_LCICoverItem <src.policies.helpers.branching_solvers.pricing.solver._LCICoverItem>`
  - ```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver._LCICoverItem
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.helpers.branching_solvers.pricing.solver.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver.logger
```

````

````{py:data} _LCICoverItem
:canonical: src.policies.helpers.branching_solvers.pricing.solver._LCICoverItem
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver._LCICoverItem
```

````

`````{py:class} RCSPPSolver(n_nodes: int, cost_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue_per_kg: float, cost_per_km: float, mandatory_nodes: typing.Optional[typing.Set[int]] = None, use_ng_routes: bool = True, ng_neighborhood_size: int = 8, ng_neighborhoods: typing.Optional[typing.Dict[int, typing.Set[int]]] = None, node_prizes: typing.Optional[typing.Dict[int, float]] = None)
:canonical: src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver

```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver.__init__
```

````{py:method} _precompute_sorted_neighbors() -> typing.Dict[int, typing.List[int]]
:canonical: src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver._precompute_sorted_neighbors

```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver._precompute_sorted_neighbors
```

````

````{py:method} _compute_ng_neighborhoods() -> typing.Dict[int, typing.Set[int]]
:canonical: src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver._compute_ng_neighborhoods

```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver._compute_ng_neighborhoods
```

````

````{py:method} save_ng_snapshot() -> typing.Dict[int, typing.Set[int]]
:canonical: src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver.save_ng_snapshot

```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver.save_ng_snapshot
```

````

````{py:method} restore_ng_snapshot(snapshot: typing.Dict[int, typing.Set[int]]) -> None
:canonical: src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver.restore_ng_snapshot

```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver.restore_ng_snapshot
```

````

````{py:method} enforce_elementarity(nodes: typing.List[int]) -> int
:canonical: src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver.enforce_elementarity

```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver.enforce_elementarity
```

````

````{py:method} expand_ng_neighborhoods(cycles: typing.List[typing.Tuple[int, ...]]) -> int
:canonical: src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver.expand_ng_neighborhoods

```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver.expand_ng_neighborhoods
```

````

````{py:method} solve(dual_values: typing.Union[typing.Dict[int, float], typing.Dict[str, typing.Any]], max_routes: int = 10, branching_constraints: typing.Optional[typing.List[logic.src.policies.helpers.branching_solvers.branching.constraints.AnyBranchingConstraint]] = None, capacity_cut_duals: typing.Optional[typing.Dict[typing.FrozenSet[int], float]] = None, sri_cut_duals: typing.Optional[typing.Dict[typing.FrozenSet[int], float]] = None, edge_clique_cut_duals: typing.Optional[typing.Dict[typing.Tuple[int, int], float]] = None, forced_nodes: typing.Optional[typing.Set[int]] = None, rf_conflicts: typing.Optional[typing.Dict[int, typing.Set[int]]] = None, is_farkas: bool = False, exact_mode: bool = False) -> typing.List[logic.src.policies.helpers.branching_solvers.common.route.Route]
:canonical: src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver.solve

```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver.solve
```

````

````{py:method} _compute_completion_bounds()
:canonical: src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver._compute_completion_bounds

```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver._compute_completion_bounds
```

````

````{py:method} _preprocess_constraints(constraints: typing.List[typing.Any])
:canonical: src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver._preprocess_constraints

```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver._preprocess_constraints
```

````

````{py:method} _label_correcting_algorithm(max_routes: int, forbidden_arcs: typing.FrozenSet[typing.Tuple[int, int]], required_successors: typing.Dict[int, int], required_predecessors: typing.Dict[int, int], rf_separate: typing.Set[typing.Tuple[int, int]], rf_together: typing.Set[typing.Tuple[int, int]], rcc_duals: typing.Dict[typing.FrozenSet[int], float], active_sri_subsets: typing.List[typing.FrozenSet[int]], sri_dual_values: typing.List[float], node_to_sri: typing.Dict[int, typing.List[int]], edge_clique_duals: typing.Dict[typing.Tuple[int, int], float], lci_cover_items: typing.List[src.policies.helpers.branching_solvers.pricing.solver._LCICoverItem], exact_mode: bool = False) -> typing.List[logic.src.policies.helpers.branching_solvers.common.route.Route]
:canonical: src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver._label_correcting_algorithm

```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver._label_correcting_algorithm
```

````

````{py:method} _get_neighbors(node: int, limit: int) -> typing.List[int]
:canonical: src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver._get_neighbors

```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver._get_neighbors
```

````

````{py:method} _extend_to_depot(label: logic.src.policies.helpers.branching_solvers.pricing.labels.Label) -> typing.Optional[logic.src.policies.helpers.branching_solvers.pricing.labels.Label]
:canonical: src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver._extend_to_depot

```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver._extend_to_depot
```

````

````{py:method} _compute_route_details(nodes: typing.List[int]) -> logic.src.policies.helpers.branching_solvers.common.route.Route
:canonical: src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver._compute_route_details

```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver._compute_route_details
```

````

````{py:method} _extend_label(label: logic.src.policies.helpers.branching_solvers.pricing.labels.Label, next_node: int, forbidden: typing.FrozenSet[typing.Tuple[int, int]], rcc_duals: typing.Dict[typing.FrozenSet[int], float], active_sri: typing.List[typing.FrozenSet[int]], sri_duals: typing.List[float], node_to_sri: typing.Dict[int, typing.List[int]], edge_clique_duals: typing.Dict[typing.Tuple[int, int], float], lci_cover_items: typing.List[src.policies.helpers.branching_solvers.pricing.solver._LCICoverItem]) -> typing.Optional[logic.src.policies.helpers.branching_solvers.pricing.labels.Label]
:canonical: src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver._extend_label

```{autodoc2-docstring} src.policies.helpers.branching_solvers.pricing.solver.RCSPPSolver._extend_label
```

````

`````
