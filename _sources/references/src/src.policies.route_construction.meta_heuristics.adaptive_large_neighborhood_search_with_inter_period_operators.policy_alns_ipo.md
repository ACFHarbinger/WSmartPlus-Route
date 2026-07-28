# {py:mod}`src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.policy_alns_ipo`

```{py:module} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.policy_alns_ipo
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.policy_alns_ipo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ALNSInterPeriodOperatorsPolicy <src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.policy_alns_ipo.ALNSInterPeriodOperatorsPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.policy_alns_ipo.ALNSInterPeriodOperatorsPolicy
    :summary:
    ```
````

### API

`````{py:class} ALNSInterPeriodOperatorsPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.ALNSIPOConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.policy_alns_ipo.ALNSInterPeriodOperatorsPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.policy_alns_ipo.ALNSInterPeriodOperatorsPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.policy_alns_ipo.ALNSInterPeriodOperatorsPolicy.__init__
```

````{py:method} _config_class() -> typing.Type[logic.src.configs.policies.ALNSIPOConfig]
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.policy_alns_ipo.ALNSInterPeriodOperatorsPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.policy_alns_ipo.ALNSInterPeriodOperatorsPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.policy_alns_ipo.ALNSInterPeriodOperatorsPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.policy_alns_ipo.ALNSInterPeriodOperatorsPolicy._get_config_key
```

````

````{py:method} _run_multi_period_solver(problem: logic.src.interfaces.context.problem_context.ProblemContext, multi_day_ctx: typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.policy_alns_ipo.ALNSInterPeriodOperatorsPolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.policy_alns_ipo.ALNSInterPeriodOperatorsPolicy._run_multi_period_solver
```

````

`````
