# {py:mod}`src.policies.branch_and_price.bp`

```{py:module} src.policies.branch_and_price.bp
```

```{autodoc2-docstring} src.policies.branch_and_price.bp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BranchAndPriceSolver <src.policies.branch_and_price.bp.BranchAndPriceSolver>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver
    :summary:
    ```
````

### API

`````{py:class} BranchAndPriceSolver(n_nodes: int, cost_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue_per_kg: float, cost_per_km: float, mandatory_nodes: typing.Optional[typing.Set[int]] = None, max_iterations: int = 100, max_routes_per_iteration: int = 10, optimality_gap: float = 0.0001, use_ryan_foster: bool = False, branching_strategy: str = 'edge', tree_search_strategy: str = 'best_first', max_branch_nodes: int = 1000, use_exact_pricing: bool = False, vehicle_limit: typing.Optional[int] = None, use_ng_routes: bool = True, ng_neighborhood_size: int = 8, cleanup_frequency: int = 20, cleanup_threshold: float = -100.0, early_termination_gap: float = 0.001, multiple_waste_types: bool = False, allow_heuristic_ryan_foster: bool = False)
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[int], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver.solve

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver.solve
```

````

````{py:method} _solve_without_branching() -> typing.Tuple[typing.List[int], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._solve_without_branching

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._solve_without_branching
```

````

````{py:method} _solve_with_branching() -> typing.Tuple[typing.List[int], typing.Optional[float], typing.Dict[str, typing.Any]]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._solve_with_branching

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._solve_with_branching
```

````

````{py:method} _solve_node(node: src.policies.branch_and_price.branching.BranchNode) -> typing.Tuple[typing.Optional[float], typing.Dict[int, float], typing.List[src.policies.branch_and_price.master_problem.Route]]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._solve_node

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._solve_node
```

````

````{py:method} _column_generation(master: src.policies.branch_and_price.master_problem.VRPPMasterProblem, pricing: typing.Tuple[src.policies.branch_and_price.pricing_subproblem.PricingSubproblem, typing.Optional[src.policies.branch_and_price.rcspp_dp.RCSPPSolver]]) -> typing.Tuple[float, typing.Dict[int, float]]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._column_generation

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._column_generation
```

````

````{py:method} _column_generation_with_constraints(master: src.policies.branch_and_price.master_problem.VRPPMasterProblem, pricing: typing.Tuple[src.policies.branch_and_price.pricing_subproblem.PricingSubproblem, typing.Optional[src.policies.branch_and_price.rcspp_dp.RCSPPSolver]], node: src.policies.branch_and_price.branching.BranchNode, constraints: typing.List[src.policies.branch_and_price.branching.AnyBranchingConstraint]) -> typing.Tuple[float, typing.Dict[int, float]]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._column_generation_with_constraints

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._column_generation_with_constraints
```

````

````{py:method} _call_pricing(pricing: typing.Any, dual_values: typing.Dict[int, float], constraints: typing.Optional[typing.List[src.policies.branch_and_price.branching.AnyBranchingConstraint]]) -> typing.List[typing.Tuple[typing.List[int], float]]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._call_pricing

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._call_pricing
```

````

````{py:method} _generate_initial_routes(pricing: typing.Any) -> typing.List[src.policies.branch_and_price.master_problem.Route]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._generate_initial_routes

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._generate_initial_routes
```

````

````{py:method} _add_routes_to_master(master: src.policies.branch_and_price.master_problem.VRPPMasterProblem, pricing: typing.Any, new_routes: typing.List[typing.Tuple[typing.List[int], float]]) -> None
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._add_routes_to_master

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._add_routes_to_master
```

````

````{py:method} _routes_to_tour(routes: typing.List[src.policies.branch_and_price.master_problem.Route]) -> typing.List[int]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._routes_to_tour

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._routes_to_tour
```

````

````{py:method} _make_master() -> src.policies.branch_and_price.master_problem.VRPPMasterProblem
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._make_master

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._make_master
```

````

````{py:method} _make_pricing() -> typing.Tuple[src.policies.branch_and_price.pricing_subproblem.PricingSubproblem, typing.Optional[src.policies.branch_and_price.rcspp_dp.RCSPPSolver]]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._make_pricing

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._make_pricing
```

````

````{py:method} _build_master_for_node(node: src.policies.branch_and_price.branching.BranchNode, routes: typing.List[src.policies.branch_and_price.master_problem.Route]) -> src.policies.branch_and_price.master_problem.VRPPMasterProblem
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._build_master_for_node

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._build_master_for_node
```

````

````{py:method} _is_integer_solution(route_values: typing.Dict[int, float], tol: float = 0.0001) -> bool
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._is_integer_solution

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._is_integer_solution
```

````

````{py:method} _get_statistics() -> typing.Dict[str, typing.Any]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._get_statistics

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._get_statistics
```

````

`````
