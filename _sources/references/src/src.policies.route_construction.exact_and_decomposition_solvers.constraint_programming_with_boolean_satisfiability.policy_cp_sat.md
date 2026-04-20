# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.policy_cp_sat`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.policy_cp_sat
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.policy_cp_sat
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CPSATPolicy <src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy
    :summary:
    ```
````

### API

`````{py:class} CPSATPolicy(config: typing.Any = None)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy.__init__
```

````{py:method} _config_class()
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy._get_config_key
```

````

````{py:method} _run_multi_period_solver(problem: logic.src.interfaces.context.problem_context.ProblemContext, multi_day_ctx: typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy._run_multi_period_solver
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.Optional[typing.List[int]] = None, **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.constraint_programming_with_boolean_satisfiability.policy_cp_sat.CPSATPolicy._run_solver
```

````

`````
