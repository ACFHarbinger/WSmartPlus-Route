# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PricingSubproblem <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem
    :summary:
    ```
````

### API

`````{py:class} PricingSubproblem(n_nodes: int, cost_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue_per_kg: float, cost_per_km: float, mandatory_nodes: typing.Set[int])
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem.__init__
```

````{py:attribute} IS_EXACT
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem.IS_EXACT
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem.IS_EXACT
```

````

````{py:method} solve(dual_values: typing.Dict[typing.Any, typing.Any], max_routes: int = 10, active_constraints: typing.Optional[typing.List[typing.Any]] = None, capacity_cut_duals: typing.Optional[typing.Dict[typing.Any, float]] = None, assert_not_exact: bool = False, **kwargs: typing.Any) -> typing.List[typing.Tuple[typing.List[int], float]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem.solve

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem.solve
```

````

````{py:method} _preprocess_constraints(constraints: typing.List[typing.Any]) -> typing.Tuple[typing.Set[typing.Tuple[int, int]], typing.Dict[int, int], typing.Dict[int, int], typing.Set[typing.Tuple[int, int]], typing.Set[typing.Tuple[int, int]]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem._preprocess_constraints

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem._preprocess_constraints
```

````

````{py:method} _evaluate_candidate_insertion(v: int, v_waste: float, route: typing.List[int], dual_values: typing.Dict[typing.Any, float], forbidden_arcs: typing.Set[typing.Tuple[int, int]], req_successors: typing.Dict[int, int], req_predecessors: typing.Dict[int, int], rf_separate_pairs: typing.Set[typing.Tuple[int, int]], visited: typing.Set[int], node_prizes: typing.Optional[typing.Dict[int, float]] = None) -> typing.Tuple[typing.Optional[int], float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem._evaluate_candidate_insertion

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem._evaluate_candidate_insertion
```

````

````{py:method} _greedy_route_construction(start_node: int, dual_values: typing.Dict[typing.Any, float], forbidden_arcs: typing.Set[typing.Tuple[int, int]], req_successors: typing.Dict[int, int], req_predecessors: typing.Dict[int, int], rf_separate_pairs: typing.Set[typing.Tuple[int, int]], rf_together_pairs: typing.Set[typing.Tuple[int, int]], capacity_cut_duals: typing.Optional[typing.Dict[typing.Any, float]] = None, node_prizes: typing.Optional[typing.Dict[int, float]] = None) -> typing.Tuple[typing.List[int], float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem._greedy_route_construction

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem._greedy_route_construction
```

````

````{py:method} _violates_rf_separation(candidate: int, visited: typing.Set[int], rf_separate_pairs: typing.Set[typing.Tuple[int, int]]) -> bool
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem._violates_rf_separation
:staticmethod:

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem._violates_rf_separation
```

````

````{py:method} _compute_reduced_cost(route: typing.List[int], dual_values: typing.Dict[int, float], capacity_cut_duals: typing.Optional[typing.Dict[typing.Any, float]] = None, node_prizes: typing.Optional[typing.Dict[int, float]] = None) -> float
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem._compute_reduced_cost

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem._compute_reduced_cost
```

````

````{py:method} compute_route_details(route: typing.List[int]) -> typing.Tuple[float, float, float, typing.Set[int]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem.compute_route_details

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price.pricing_subproblem.PricingSubproblem.compute_route_details
```

````

`````
