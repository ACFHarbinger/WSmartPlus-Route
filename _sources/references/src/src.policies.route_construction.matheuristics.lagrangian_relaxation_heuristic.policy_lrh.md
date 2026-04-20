# {py:mod}`src.policies.route_construction.matheuristics.lagrangian_relaxation_heuristic.policy_lrh`

```{py:module} src.policies.route_construction.matheuristics.lagrangian_relaxation_heuristic.policy_lrh
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lagrangian_relaxation_heuristic.policy_lrh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LagrangianRelaxationHeuristicPolicy <src.policies.route_construction.matheuristics.lagrangian_relaxation_heuristic.policy_lrh.LagrangianRelaxationHeuristicPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.lagrangian_relaxation_heuristic.policy_lrh.LagrangianRelaxationHeuristicPolicy
    :summary:
    ```
````

### API

`````{py:class} LagrangianRelaxationHeuristicPolicy(config: typing.Any = None)
:canonical: src.policies.route_construction.matheuristics.lagrangian_relaxation_heuristic.policy_lrh.LagrangianRelaxationHeuristicPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lagrangian_relaxation_heuristic.policy_lrh.LagrangianRelaxationHeuristicPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lagrangian_relaxation_heuristic.policy_lrh.LagrangianRelaxationHeuristicPolicy.__init__
```

````{py:method} _solve_relaxed_subproblem(problem: logic.src.interfaces.context.problem_context.ProblemContext, lambdas: typing.List[float]) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.matheuristics.lagrangian_relaxation_heuristic.policy_lrh.LagrangianRelaxationHeuristicPolicy._solve_relaxed_subproblem

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lagrangian_relaxation_heuristic.policy_lrh.LagrangianRelaxationHeuristicPolicy._solve_relaxed_subproblem
```

````

````{py:method} _repair_to_feasible(problem: logic.src.interfaces.context.problem_context.ProblemContext, plan: typing.List[typing.List[typing.List[int]]]) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.matheuristics.lagrangian_relaxation_heuristic.policy_lrh.LagrangianRelaxationHeuristicPolicy._repair_to_feasible

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lagrangian_relaxation_heuristic.policy_lrh.LagrangianRelaxationHeuristicPolicy._repair_to_feasible
```

````

````{py:method} _evaluate(plan, problem)
:canonical: src.policies.route_construction.matheuristics.lagrangian_relaxation_heuristic.policy_lrh.LagrangianRelaxationHeuristicPolicy._evaluate

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lagrangian_relaxation_heuristic.policy_lrh.LagrangianRelaxationHeuristicPolicy._evaluate
```

````

````{py:method} _run_multi_period_solver(problem: logic.src.interfaces.context.problem_context.ProblemContext, multi_day_ctx: typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.matheuristics.lagrangian_relaxation_heuristic.policy_lrh.LagrangianRelaxationHeuristicPolicy._run_multi_period_solver

````

`````
