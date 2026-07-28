# {py:mod}`src.policies.route_construction.hyper_heuristics.matheuristic_hyper_heuristic.policy_mhh`

```{py:module} src.policies.route_construction.hyper_heuristics.matheuristic_hyper_heuristic.policy_mhh
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.matheuristic_hyper_heuristic.policy_mhh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MHHPolicy <src.policies.route_construction.hyper_heuristics.matheuristic_hyper_heuristic.policy_mhh.MHHPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.matheuristic_hyper_heuristic.policy_mhh.MHHPolicy
    :summary:
    ```
````

### API

`````{py:class} MHHPolicy(config: typing.Any = None)
:canonical: src.policies.route_construction.hyper_heuristics.matheuristic_hyper_heuristic.policy_mhh.MHHPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.matheuristic_hyper_heuristic.policy_mhh.MHHPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.matheuristic_hyper_heuristic.policy_mhh.MHHPolicy.__init__
```

````{py:method} _evaluate(plan: typing.List[typing.List[typing.List[int]]], problem: logic.src.interfaces.context.problem_context.ProblemContext) -> float
:canonical: src.policies.route_construction.hyper_heuristics.matheuristic_hyper_heuristic.policy_mhh.MHHPolicy._evaluate

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.matheuristic_hyper_heuristic.policy_mhh.MHHPolicy._evaluate
```

````

````{py:method} _run_multi_period_solver(problem: logic.src.interfaces.context.problem_context.ProblemContext, multi_day_ctx: typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.hyper_heuristics.matheuristic_hyper_heuristic.policy_mhh.MHHPolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.matheuristic_hyper_heuristic.policy_mhh.MHHPolicy._run_multi_period_solver
```

````

`````
