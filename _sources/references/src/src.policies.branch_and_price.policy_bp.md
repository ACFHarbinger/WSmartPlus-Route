# {py:mod}`src.policies.branch_and_price.policy_bp`

```{py:module} src.policies.branch_and_price.policy_bp
```

```{autodoc2-docstring} src.policies.branch_and_price.policy_bp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PolicyBP <src.policies.branch_and_price.policy_bp.PolicyBP>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.policy_bp.PolicyBP
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_branch_and_price <src.policies.branch_and_price.policy_bp.run_branch_and_price>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.policy_bp.run_branch_and_price
    :summary:
    ```
````

### API

`````{py:class} PolicyBP(max_iterations: int = 100, max_routes_per_iteration: int = 10, optimality_gap: float = 0.0001, use_ryan_foster_branching: bool = False, max_branch_nodes: int = 1000, use_exact_pricing: bool = False, **kwargs: typing.Any)
:canonical: src.policies.branch_and_price.policy_bp.PolicyBP

```{autodoc2-docstring} src.policies.branch_and_price.policy_bp.PolicyBP
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price.policy_bp.PolicyBP.__init__
```

````{py:method} solve(cost_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue_per_kg: float, cost_per_km: float, mandatory_nodes: typing.Optional[typing.List[int]] = None) -> typing.Dict[str, typing.Any]
:canonical: src.policies.branch_and_price.policy_bp.PolicyBP.solve

```{autodoc2-docstring} src.policies.branch_and_price.policy_bp.PolicyBP.solve
```

````

````{py:method} __call__(*args: typing.Any, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]
:canonical: src.policies.branch_and_price.policy_bp.PolicyBP.__call__

```{autodoc2-docstring} src.policies.branch_and_price.policy_bp.PolicyBP.__call__
```

````

`````

````{py:function} run_branch_and_price(cost_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue_per_kg: float, cost_per_km: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, max_iterations: int = 100, max_routes_per_iteration: int = 10, optimality_gap: float = 0.0001, use_ryan_foster_branching: bool = False, max_branch_nodes: int = 1000) -> typing.Dict[str, typing.Any]
:canonical: src.policies.branch_and_price.policy_bp.run_branch_and_price

```{autodoc2-docstring} src.policies.branch_and_price.policy_bp.run_branch_and_price
```
````
