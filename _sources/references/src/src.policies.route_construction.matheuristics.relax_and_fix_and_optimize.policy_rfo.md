# {py:mod}`src.policies.route_construction.matheuristics.relax_and_fix_and_optimize.policy_rfo`

```{py:module} src.policies.route_construction.matheuristics.relax_and_fix_and_optimize.policy_rfo
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.relax_and_fix_and_optimize.policy_rfo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RelaxFixOptimizePolicy <src.policies.route_construction.matheuristics.relax_and_fix_and_optimize.policy_rfo.RelaxFixOptimizePolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.relax_and_fix_and_optimize.policy_rfo.RelaxFixOptimizePolicy
    :summary:
    ```
````

### API

`````{py:class} RelaxFixOptimizePolicy(config: typing.Any = None)
:canonical: src.policies.route_construction.matheuristics.relax_and_fix_and_optimize.policy_rfo.RelaxFixOptimizePolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.matheuristics.relax_and_fix_and_optimize.policy_rfo.RelaxFixOptimizePolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.relax_and_fix_and_optimize.policy_rfo.RelaxFixOptimizePolicy.__init__
```

````{py:method} _setup_multi_day_model(problem: logic.src.interfaces.context.problem_context.ProblemContext, int_days: set, relaxed_days: set, fixed_plan: typing.Dict[int, typing.List[int]])
:canonical: src.policies.route_construction.matheuristics.relax_and_fix_and_optimize.policy_rfo.RelaxFixOptimizePolicy._setup_multi_day_model

```{autodoc2-docstring} src.policies.route_construction.matheuristics.relax_and_fix_and_optimize.policy_rfo.RelaxFixOptimizePolicy._setup_multi_day_model
```

````

````{py:method} _run_multi_period_solver(problem: logic.src.interfaces.context.problem_context.ProblemContext, multi_day_ctx: typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.matheuristics.relax_and_fix_and_optimize.policy_rfo.RelaxFixOptimizePolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.matheuristics.relax_and_fix_and_optimize.policy_rfo.RelaxFixOptimizePolicy._run_multi_period_solver
```

````

`````
