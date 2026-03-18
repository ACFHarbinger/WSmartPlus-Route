# {py:mod}`src.policies.branch_and_price.pricing_subproblem`

```{py:module} src.policies.branch_and_price.pricing_subproblem
```

```{autodoc2-docstring} src.policies.branch_and_price.pricing_subproblem
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PricingSubproblem <src.policies.branch_and_price.pricing_subproblem.PricingSubproblem>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.pricing_subproblem.PricingSubproblem
    :summary:
    ```
````

### API

`````{py:class} PricingSubproblem(n_nodes: int, cost_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue_per_kg: float, cost_per_km: float, mandatory_nodes: typing.Set[int])
:canonical: src.policies.branch_and_price.pricing_subproblem.PricingSubproblem

```{autodoc2-docstring} src.policies.branch_and_price.pricing_subproblem.PricingSubproblem
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price.pricing_subproblem.PricingSubproblem.__init__
```

````{py:method} solve(dual_values: typing.Dict[int, float], max_routes: int = 10) -> typing.List[typing.Tuple[typing.List[int], float]]
:canonical: src.policies.branch_and_price.pricing_subproblem.PricingSubproblem.solve

```{autodoc2-docstring} src.policies.branch_and_price.pricing_subproblem.PricingSubproblem.solve
```

````

````{py:method} _greedy_route_construction(start_node: int, dual_values: typing.Dict[int, float]) -> typing.Tuple[typing.List[int], float]
:canonical: src.policies.branch_and_price.pricing_subproblem.PricingSubproblem._greedy_route_construction

```{autodoc2-docstring} src.policies.branch_and_price.pricing_subproblem.PricingSubproblem._greedy_route_construction
```

````

````{py:method} _compute_reduced_cost(route: typing.List[int], dual_values: typing.Dict[int, float]) -> float
:canonical: src.policies.branch_and_price.pricing_subproblem.PricingSubproblem._compute_reduced_cost

```{autodoc2-docstring} src.policies.branch_and_price.pricing_subproblem.PricingSubproblem._compute_reduced_cost
```

````

````{py:method} compute_route_details(route: typing.List[int]) -> typing.Tuple[float, float, float, typing.Set[int]]
:canonical: src.policies.branch_and_price.pricing_subproblem.PricingSubproblem.compute_route_details

```{autodoc2-docstring} src.policies.branch_and_price.pricing_subproblem.PricingSubproblem.compute_route_details
```

````

`````
