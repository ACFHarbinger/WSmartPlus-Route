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

* - {py:obj}`BranchAndPricePolicy <src.policies.branch_and_price.policy_bp.BranchAndPricePolicy>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.policy_bp.BranchAndPricePolicy
    :summary:
    ```
````

### API

`````{py:class} BranchAndPricePolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.BPConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.branch_and_price.policy_bp.BranchAndPricePolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.branch_and_price.policy_bp.BranchAndPricePolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price.policy_bp.BranchAndPricePolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.branch_and_price.policy_bp.BranchAndPricePolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.branch_and_price.policy_bp.BranchAndPricePolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.branch_and_price.policy_bp.BranchAndPricePolicy._get_config_key

```{autodoc2-docstring} src.policies.branch_and_price.policy_bp.BranchAndPricePolicy._get_config_key
```

````

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.Union[typing.List[int], typing.List[typing.List[int]]], float, typing.Any]
:canonical: src.policies.branch_and_price.policy_bp.BranchAndPricePolicy.execute

```{autodoc2-docstring} src.policies.branch_and_price.policy_bp.BranchAndPricePolicy.execute
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.branch_and_price.policy_bp.BranchAndPricePolicy._run_solver

```{autodoc2-docstring} src.policies.branch_and_price.policy_bp.BranchAndPricePolicy._run_solver
```

````

`````
