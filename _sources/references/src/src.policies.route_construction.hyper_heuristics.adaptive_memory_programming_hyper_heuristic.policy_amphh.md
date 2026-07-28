# {py:mod}`src.policies.route_construction.hyper_heuristics.adaptive_memory_programming_hyper_heuristic.policy_amphh`

```{py:module} src.policies.route_construction.hyper_heuristics.adaptive_memory_programming_hyper_heuristic.policy_amphh
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.adaptive_memory_programming_hyper_heuristic.policy_amphh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AMPHHPolicy <src.policies.route_construction.hyper_heuristics.adaptive_memory_programming_hyper_heuristic.policy_amphh.AMPHHPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.adaptive_memory_programming_hyper_heuristic.policy_amphh.AMPHHPolicy
    :summary:
    ```
````

### API

`````{py:class} AMPHHPolicy(config: typing.Any = None)
:canonical: src.policies.route_construction.hyper_heuristics.adaptive_memory_programming_hyper_heuristic.policy_amphh.AMPHHPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.adaptive_memory_programming_hyper_heuristic.policy_amphh.AMPHHPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.adaptive_memory_programming_hyper_heuristic.policy_amphh.AMPHHPolicy.__init__
```

````{py:method} _evaluate(plan: typing.List[typing.List[typing.List[int]]], problem: logic.src.interfaces.context.problem_context.ProblemContext) -> float
:canonical: src.policies.route_construction.hyper_heuristics.adaptive_memory_programming_hyper_heuristic.policy_amphh.AMPHHPolicy._evaluate

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.adaptive_memory_programming_hyper_heuristic.policy_amphh.AMPHHPolicy._evaluate
```

````

````{py:method} _update_memory(prof: float, plan: typing.List[typing.List[typing.List[int]]])
:canonical: src.policies.route_construction.hyper_heuristics.adaptive_memory_programming_hyper_heuristic.policy_amphh.AMPHHPolicy._update_memory

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.adaptive_memory_programming_hyper_heuristic.policy_amphh.AMPHHPolicy._update_memory
```

````

````{py:method} _run_multi_period_solver(problem: logic.src.interfaces.context.problem_context.ProblemContext, multi_day_ctx: typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.hyper_heuristics.adaptive_memory_programming_hyper_heuristic.policy_amphh.AMPHHPolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.adaptive_memory_programming_hyper_heuristic.policy_amphh.AMPHHPolicy._run_multi_period_solver
```

````

`````
