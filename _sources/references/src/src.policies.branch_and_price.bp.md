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

`````{py:class} BranchAndPriceSolver(n_nodes: int, cost_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue_per_kg: float, cost_per_km: float, mandatory_nodes: typing.Optional[typing.Set[int]] = None, max_iterations: int = 100, max_routes_per_iteration: int = 10, optimality_gap: float = 0.0001, use_ryan_foster: bool = False, branching_strategy: str = 'edge', tree_search_strategy: str = 'best_first', max_branch_nodes: int = 1000, use_exact_pricing: bool = False, vehicle_limit: typing.Optional[int] = None, use_ng_routes: bool = True, ng_neighborhood_size: int = 8, cleanup_frequency: int = 20, cleanup_threshold: float = -100.0, early_termination_gap: float = 0.001, multiple_waste_types: bool = False, allow_heuristic_ryan_foster: bool = False, params: typing.Optional[src.policies.branch_and_price.params.BPParams] = None)
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

````{py:method} _solve_node(node: logic.src.policies.other.branching_solvers.BranchNode, parent_routes: typing.Optional[typing.List[logic.src.policies.other.branching_solvers.Route]] = None) -> typing.Tuple[typing.Optional[float], typing.Dict[int, float], typing.List[logic.src.policies.other.branching_solvers.Route]]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._solve_node

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._solve_node
```

````

````{py:method} _column_generation(master: logic.src.policies.other.branching_solvers.VRPPMasterProblem, pricing: typing.Tuple[src.policies.branch_and_price.pricing_subproblem.PricingSubproblem, typing.Optional[logic.src.policies.other.branching_solvers.RCSPPSolver]]) -> typing.Tuple[float, typing.Dict[int, float]]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._column_generation

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._column_generation
```

````

````{py:method} _column_generation_with_constraints(master: logic.src.policies.other.branching_solvers.VRPPMasterProblem, pricing: typing.Tuple[src.policies.branch_and_price.pricing_subproblem.PricingSubproblem, typing.Optional[logic.src.policies.other.branching_solvers.RCSPPSolver]], node: logic.src.policies.other.branching_solvers.BranchNode, constraints: typing.List[logic.src.policies.other.branching_solvers.AnyBranchingConstraint]) -> typing.Tuple[float, typing.Dict[int, float], float]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._column_generation_with_constraints

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._column_generation_with_constraints
```

````

````{py:method} _call_pricing(pricing: typing.Union[src.policies.branch_and_price.pricing_subproblem.PricingSubproblem, logic.src.policies.other.branching_solvers.RCSPPSolver], dual_values: typing.Dict[typing.Union[int, frozenset[int], str, typing.Tuple[int, int]], float], constraints: typing.Optional[typing.List[logic.src.policies.other.branching_solvers.AnyBranchingConstraint]]) -> typing.List[typing.Tuple[typing.List[int], float]]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._call_pricing

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._call_pricing
```

````

````{py:method} _generate_initial_routes(pricing: typing.Any) -> typing.List[logic.src.policies.other.branching_solvers.Route]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._generate_initial_routes

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._generate_initial_routes
```

````

````{py:method} _add_routes_to_master(master: logic.src.policies.other.branching_solvers.VRPPMasterProblem, pricing: typing.Any, new_routes: typing.List[typing.Tuple[typing.List[int], float]]) -> None
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._add_routes_to_master

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._add_routes_to_master
```

````

````{py:method} _routes_to_tour(routes: typing.List[logic.src.policies.other.branching_solvers.Route]) -> typing.List[int]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._routes_to_tour

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._routes_to_tour
```

````

````{py:method} _make_master() -> logic.src.policies.other.branching_solvers.VRPPMasterProblem
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._make_master

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._make_master
```

````

````{py:method} _make_pricing() -> typing.Tuple[src.policies.branch_and_price.pricing_subproblem.PricingSubproblem, typing.Optional[logic.src.policies.other.branching_solvers.RCSPPSolver]]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._make_pricing

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._make_pricing
```

````

````{py:method} _build_master_for_node(node: logic.src.policies.other.branching_solvers.BranchNode, routes: typing.List[logic.src.policies.other.branching_solvers.Route]) -> logic.src.policies.other.branching_solvers.VRPPMasterProblem
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._build_master_for_node

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._build_master_for_node
```

````

````{py:method} _is_integer_solution(route_values: typing.Dict[int, float], tol: float = 1e-05) -> bool
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._is_integer_solution

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._is_integer_solution
```

````

````{py:method} _resolve_branching_strategy(branching_strategy: str, use_ryan_foster: bool, values: typing.Dict[str, typing.Any]) -> str
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._resolve_branching_strategy
:staticmethod:

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._resolve_branching_strategy
```

````

````{py:method} _get_statistics() -> typing.Dict[str, typing.Any]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._get_statistics

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._get_statistics
```

````

`````
