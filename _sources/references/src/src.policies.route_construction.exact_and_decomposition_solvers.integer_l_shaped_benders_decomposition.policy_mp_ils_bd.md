# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_mp_ils_bd`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_mp_ils_bd
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_mp_ils_bd
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MPIntegerLShapedPolicy <src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_mp_ils_bd.MPIntegerLShapedPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_mp_ils_bd.MPIntegerLShapedPolicy
    :summary:
    ```
````

### API

`````{py:class} MPIntegerLShapedPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.MPILSBDConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_mp_ils_bd.MPIntegerLShapedPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_mp_ils_bd.MPIntegerLShapedPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_mp_ils_bd.MPIntegerLShapedPolicy.__init__
```

````{py:method} _config_class() -> typing.Type[logic.src.configs.policies.MPILSBDConfig]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_mp_ils_bd.MPIntegerLShapedPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_mp_ils_bd.MPIntegerLShapedPolicy._get_config_key

````

````{py:method} _run_multi_period_solver(tree: typing.Any, capacity: float, revenue: float, cost_unit: float, **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_mp_ils_bd.MPIntegerLShapedPolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.policy_mp_ils_bd.MPIntegerLShapedPolicy._run_multi_period_solver
```

````

`````
