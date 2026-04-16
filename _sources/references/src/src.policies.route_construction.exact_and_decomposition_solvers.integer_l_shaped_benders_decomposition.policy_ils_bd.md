# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_ils_bd`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_ils_bd
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_ils_bd
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IntegerLShapedPolicy <src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy
    :summary:
    ```
````

### API

`````{py:class} IntegerLShapedPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.IntegerLShapedBendersConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy.__init__
```

````{py:method} _config_class() -> typing.Type[logic.src.configs.policies.IntegerLShapedBendersConfig]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy._get_config_key
```

````

````{py:method} _run_multi_period_solver(tree: typing.Any, capacity: float, revenue: float, cost_unit: float, **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy._run_multi_period_solver
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy._run_solver
```

````

`````
