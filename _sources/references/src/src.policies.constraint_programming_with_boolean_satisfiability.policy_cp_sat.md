# {py:mod}`src.policies.constraint_programming_with_boolean_satisfiability.policy_cp_sat`

```{py:module} src.policies.constraint_programming_with_boolean_satisfiability.policy_cp_sat
```

```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.policy_cp_sat
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CPSATPolicy <src.policies.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy>`
  - ```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy
    :summary:
    ```
````

### API

`````{py:class} CPSATPolicy(config: typing.Any = None)
:canonical: src.policies.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy.__init__
```

````{py:method} _config_class()
:canonical: src.policies.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy._run_solver

```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy._run_solver
```

````

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy.execute

```{autodoc2-docstring} src.policies.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy.execute
```

````

`````
