# {py:mod}`src.policies.route_construction.hyper_heuristics.selection_hyper_heuristic.policy_shh`

```{py:module} src.policies.route_construction.hyper_heuristics.selection_hyper_heuristic.policy_shh
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.selection_hyper_heuristic.policy_shh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SelectionHHPolicy <src.policies.route_construction.hyper_heuristics.selection_hyper_heuristic.policy_shh.SelectionHHPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.selection_hyper_heuristic.policy_shh.SelectionHHPolicy
    :summary:
    ```
````

### API

`````{py:class} SelectionHHPolicy(config: typing.Any = None)
:canonical: src.policies.route_construction.hyper_heuristics.selection_hyper_heuristic.policy_shh.SelectionHHPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.selection_hyper_heuristic.policy_shh.SelectionHHPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.selection_hyper_heuristic.policy_shh.SelectionHHPolicy.__init__
```

````{py:method} _evaluate(plan: typing.List[typing.List[typing.List[int]]], problem: logic.src.interfaces.context.problem_context.ProblemContext) -> float
:canonical: src.policies.route_construction.hyper_heuristics.selection_hyper_heuristic.policy_shh.SelectionHHPolicy._evaluate

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.selection_hyper_heuristic.policy_shh.SelectionHHPolicy._evaluate
```

````

````{py:method} _run_multi_period_solver(problem: logic.src.interfaces.context.problem_context.ProblemContext, multi_day_ctx: typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.hyper_heuristics.selection_hyper_heuristic.policy_shh.SelectionHHPolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.selection_hyper_heuristic.policy_shh.SelectionHHPolicy._run_multi_period_solver
```

````

`````
