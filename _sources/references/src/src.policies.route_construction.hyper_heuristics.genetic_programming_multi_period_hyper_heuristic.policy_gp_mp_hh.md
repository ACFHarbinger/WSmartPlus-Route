# {py:mod}`src.policies.route_construction.hyper_heuristics.genetic_programming_multi_period_hyper_heuristic.policy_gp_mp_hh`

```{py:module} src.policies.route_construction.hyper_heuristics.genetic_programming_multi_period_hyper_heuristic.policy_gp_mp_hh
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_multi_period_hyper_heuristic.policy_gp_mp_hh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GPMPHHPolicy <src.policies.route_construction.hyper_heuristics.genetic_programming_multi_period_hyper_heuristic.policy_gp_mp_hh.GPMPHHPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_multi_period_hyper_heuristic.policy_gp_mp_hh.GPMPHHPolicy
    :summary:
    ```
````

### API

`````{py:class} GPMPHHPolicy(config: typing.Any = None)
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_multi_period_hyper_heuristic.policy_gp_mp_hh.GPMPHHPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_multi_period_hyper_heuristic.policy_gp_mp_hh.GPMPHHPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_multi_period_hyper_heuristic.policy_gp_mp_hh.GPMPHHPolicy.__init__
```

````{py:method} _evaluate(plan: typing.List[typing.List[typing.List[int]]], problem: logic.src.interfaces.context.problem_context.ProblemContext) -> float
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_multi_period_hyper_heuristic.policy_gp_mp_hh.GPMPHHPolicy._evaluate

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_multi_period_hyper_heuristic.policy_gp_mp_hh.GPMPHHPolicy._evaluate
```

````

````{py:method} _run_multi_period_solver(problem: logic.src.interfaces.context.problem_context.ProblemContext, multi_day_ctx: typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.hyper_heuristics.genetic_programming_multi_period_hyper_heuristic.policy_gp_mp_hh.GPMPHHPolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.genetic_programming_multi_period_hyper_heuristic.policy_gp_mp_hh.GPMPHHPolicy._run_multi_period_solver
```

````

`````
