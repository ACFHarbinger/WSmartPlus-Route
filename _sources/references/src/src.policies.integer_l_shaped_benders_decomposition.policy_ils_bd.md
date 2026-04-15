# {py:mod}`src.policies.integer_l_shaped_benders_decomposition.policy_ils_bd`

```{py:module} src.policies.integer_l_shaped_benders_decomposition.policy_ils_bd
```

```{autodoc2-docstring} src.policies.integer_l_shaped_benders_decomposition.policy_ils_bd
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IntegerLShapedPolicy <src.policies.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy>`
  - ```{autodoc2-docstring} src.policies.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy
    :summary:
    ```
````

### API

`````{py:class} IntegerLShapedPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.IntegerLShapedBendersConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy._get_config_key

```{autodoc2-docstring} src.policies.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy._run_solver

```{autodoc2-docstring} src.policies.integer_l_shaped_benders_decomposition.policy_ils_bd.IntegerLShapedPolicy._run_solver
```

````

`````
