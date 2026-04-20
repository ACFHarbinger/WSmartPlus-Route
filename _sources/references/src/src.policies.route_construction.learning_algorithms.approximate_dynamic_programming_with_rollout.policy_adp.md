# {py:mod}`src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.policy_adp`

```{py:module} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.policy_adp
```

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.policy_adp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ADPRolloutPolicy <src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.policy_adp.ADPRolloutPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.policy_adp.ADPRolloutPolicy
    :summary:
    ```
````

### API

`````{py:class} ADPRolloutPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.ADPRolloutConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.policy_adp.ADPRolloutPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.policy_adp.ADPRolloutPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.policy_adp.ADPRolloutPolicy.__init__
```

````{py:method} _config_class() -> typing.Type[logic.src.configs.policies.ADPRolloutConfig]
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.policy_adp.ADPRolloutPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.policy_adp.ADPRolloutPolicy._get_config_key

````

````{py:method} _run_multi_period_solver(problem: logic.src.interfaces.context.problem_context.ProblemContext, multi_day_ctx: typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.policy_adp.ADPRolloutPolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.policy_adp.ADPRolloutPolicy._run_multi_period_solver
```

````

`````
