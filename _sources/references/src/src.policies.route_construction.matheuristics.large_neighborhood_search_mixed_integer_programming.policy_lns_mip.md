# {py:mod}`src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.policy_lns_mip`

```{py:module} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.policy_lns_mip
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.policy_lns_mip
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LNSMIPPolicy <src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.policy_lns_mip.LNSMIPPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.policy_lns_mip.LNSMIPPolicy
    :summary:
    ```
````

### API

`````{py:class} LNSMIPPolicy(config: typing.Any = None)
:canonical: src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.policy_lns_mip.LNSMIPPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.policy_lns_mip.LNSMIPPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.policy_lns_mip.LNSMIPPolicy.__init__
```

````{py:method} _evaluate_plan(plan: typing.List[typing.List[typing.List[int]]], problem: logic.src.interfaces.context.problem_context.ProblemContext) -> float
:canonical: src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.policy_lns_mip.LNSMIPPolicy._evaluate_plan

```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.policy_lns_mip.LNSMIPPolicy._evaluate_plan
```

````

````{py:method} _destroy(problem: logic.src.interfaces.context.problem_context.ProblemContext, plan: typing.List[typing.List[typing.List[int]]]) -> typing.Dict[int, typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.policy_lns_mip.LNSMIPPolicy._destroy

```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.policy_lns_mip.LNSMIPPolicy._destroy
```

````

````{py:method} _repair_mip(problem: logic.src.interfaces.context.problem_context.ProblemContext, plan: typing.List[typing.List[typing.List[int]]], destroyed: typing.Dict[int, typing.List[int]]) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.policy_lns_mip.LNSMIPPolicy._repair_mip

```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.policy_lns_mip.LNSMIPPolicy._repair_mip
```

````

````{py:method} _accept(best_profit: float, new_profit: float, temp: float) -> bool
:canonical: src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.policy_lns_mip.LNSMIPPolicy._accept

```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.policy_lns_mip.LNSMIPPolicy._accept
```

````

````{py:method} _run_multi_period_solver(problem: logic.src.interfaces.context.problem_context.ProblemContext, multi_day_ctx: typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.policy_lns_mip.LNSMIPPolicy._run_multi_period_solver

````

`````
