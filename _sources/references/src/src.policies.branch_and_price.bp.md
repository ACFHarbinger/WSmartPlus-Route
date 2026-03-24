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

`````{py:class} BranchAndPriceSolver(n_nodes: int, cost_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue_per_kg: float, cost_per_km: float, mandatory_nodes: typing.Optional[typing.Set[int]] = None, max_iterations: int = 100, max_routes_per_iteration: int = 10, optimality_gap: float = 0.0001, use_ryan_foster: bool = True, max_branch_nodes: int = 1000, use_exact_pricing: bool = False)
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[int], float, typing.Dict]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver.solve

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver.solve
```

````

````{py:method} _solve_without_branching() -> typing.Tuple[typing.List[int], float, typing.Dict]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._solve_without_branching

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._solve_without_branching
```

````

````{py:method} _solve_with_branching() -> typing.Tuple[typing.List[int], typing.Optional[float], typing.Dict]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._solve_with_branching

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._solve_with_branching
```

````

````{py:method} _solve_node(node: src.policies.branch_and_price.ryan_foster_branching.BranchNode) -> typing.Tuple[typing.Optional[float], typing.Dict[int, float], typing.List[src.policies.branch_and_price.master_problem.Route]]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._solve_node

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._solve_node
```

````

````{py:method} _build_master_for_node(node: src.policies.branch_and_price.ryan_foster_branching.BranchNode, routes: typing.List[src.policies.branch_and_price.master_problem.Route]) -> src.policies.branch_and_price.master_problem.VRPPMasterProblem
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._build_master_for_node

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._build_master_for_node
```

````

````{py:method} _column_generation_with_constraints(master: src.policies.branch_and_price.master_problem.VRPPMasterProblem, pricing, node: src.policies.branch_and_price.ryan_foster_branching.BranchNode, constraints: typing.List) -> typing.Tuple[float, typing.Dict[int, float]]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._column_generation_with_constraints

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._column_generation_with_constraints
```

````

````{py:method} _generate_initial_routes(pricing) -> typing.List[src.policies.branch_and_price.master_problem.Route]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._generate_initial_routes

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._generate_initial_routes
```

````

````{py:method} _column_generation(master: src.policies.branch_and_price.master_problem.VRPPMasterProblem, pricing) -> typing.Tuple[float, typing.Dict[int, float]]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._column_generation

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._column_generation
```

````

````{py:method} _is_integer_solution(route_values: typing.Dict[int, float], tol: float = 0.0001) -> bool
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._is_integer_solution

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._is_integer_solution
```

````

````{py:method} _routes_to_tour(routes: typing.List[src.policies.branch_and_price.master_problem.Route]) -> typing.List[int]
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._routes_to_tour

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._routes_to_tour
```

````

````{py:method} _get_statistics() -> typing.Dict
:canonical: src.policies.branch_and_price.bp.BranchAndPriceSolver._get_statistics

```{autodoc2-docstring} src.policies.branch_and_price.bp.BranchAndPriceSolver._get_statistics
```

````

`````
