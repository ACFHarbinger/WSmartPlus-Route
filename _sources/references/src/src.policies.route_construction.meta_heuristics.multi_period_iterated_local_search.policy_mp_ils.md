# {py:mod}`src.policies.route_construction.meta_heuristics.multi_period_iterated_local_search.policy_mp_ils`

```{py:module} src.policies.route_construction.meta_heuristics.multi_period_iterated_local_search.policy_mp_ils
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_iterated_local_search.policy_mp_ils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MultiPeriodILSPolicy <src.policies.route_construction.meta_heuristics.multi_period_iterated_local_search.policy_mp_ils.MultiPeriodILSPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_iterated_local_search.policy_mp_ils.MultiPeriodILSPolicy
    :summary:
    ```
````

### API

`````{py:class} MultiPeriodILSPolicy(config: typing.Any = None)
:canonical: src.policies.route_construction.meta_heuristics.multi_period_iterated_local_search.policy_mp_ils.MultiPeriodILSPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_iterated_local_search.policy_mp_ils.MultiPeriodILSPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_iterated_local_search.policy_mp_ils.MultiPeriodILSPolicy.__init__
```

````{py:method} _evaluate(plan: typing.List[typing.List[typing.List[int]]], problem: logic.src.interfaces.context.problem_context.ProblemContext) -> float
:canonical: src.policies.route_construction.meta_heuristics.multi_period_iterated_local_search.policy_mp_ils.MultiPeriodILSPolicy._evaluate

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_iterated_local_search.policy_mp_ils.MultiPeriodILSPolicy._evaluate
```

````

````{py:method} _local_search(plan: typing.List[typing.List[typing.List[int]]], problem: logic.src.interfaces.context.problem_context.ProblemContext) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.meta_heuristics.multi_period_iterated_local_search.policy_mp_ils.MultiPeriodILSPolicy._local_search

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_iterated_local_search.policy_mp_ils.MultiPeriodILSPolicy._local_search
```

````

````{py:method} _perturb(plan: typing.List[typing.List[typing.List[int]]], problem: logic.src.interfaces.context.problem_context.ProblemContext) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.meta_heuristics.multi_period_iterated_local_search.policy_mp_ils.MultiPeriodILSPolicy._perturb

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_iterated_local_search.policy_mp_ils.MultiPeriodILSPolicy._perturb
```

````

````{py:method} _run_multi_period_solver(problem: logic.src.interfaces.context.problem_context.ProblemContext, multi_day_ctx: typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.meta_heuristics.multi_period_iterated_local_search.policy_mp_ils.MultiPeriodILSPolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_iterated_local_search.policy_mp_ils.MultiPeriodILSPolicy._run_multi_period_solver
```

````

`````
