# {py:mod}`src.policies.route_construction.matheuristics.column_generation_heuristic.policy_cgh`

```{py:module} src.policies.route_construction.matheuristics.column_generation_heuristic.policy_cgh
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.column_generation_heuristic.policy_cgh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ColumnGenerationHeuristicPolicy <src.policies.route_construction.matheuristics.column_generation_heuristic.policy_cgh.ColumnGenerationHeuristicPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.column_generation_heuristic.policy_cgh.ColumnGenerationHeuristicPolicy
    :summary:
    ```
````

### API

`````{py:class} ColumnGenerationHeuristicPolicy(config: typing.Any = None)
:canonical: src.policies.route_construction.matheuristics.column_generation_heuristic.policy_cgh.ColumnGenerationHeuristicPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.matheuristics.column_generation_heuristic.policy_cgh.ColumnGenerationHeuristicPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.column_generation_heuristic.policy_cgh.ColumnGenerationHeuristicPolicy.__init__
```

````{py:method} _generate_columns_heuristically(problem: logic.src.interfaces.context.problem_context.ProblemContext, duals: typing.Dict[int, float], n_routes: int, rng: numpy.random.Generator) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.column_generation_heuristic.policy_cgh.ColumnGenerationHeuristicPolicy._generate_columns_heuristically

```{autodoc2-docstring} src.policies.route_construction.matheuristics.column_generation_heuristic.policy_cgh.ColumnGenerationHeuristicPolicy._generate_columns_heuristically
```

````

````{py:method} _solve_cg_for_day(problem: logic.src.interfaces.context.problem_context.ProblemContext, rng: numpy.random.Generator) -> typing.List[int]
:canonical: src.policies.route_construction.matheuristics.column_generation_heuristic.policy_cgh.ColumnGenerationHeuristicPolicy._solve_cg_for_day

```{autodoc2-docstring} src.policies.route_construction.matheuristics.column_generation_heuristic.policy_cgh.ColumnGenerationHeuristicPolicy._solve_cg_for_day
```

````

````{py:method} _run_multi_period_solver(problem: logic.src.interfaces.context.problem_context.ProblemContext, multi_day_ctx: typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.matheuristics.column_generation_heuristic.policy_cgh.ColumnGenerationHeuristicPolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.matheuristics.column_generation_heuristic.policy_cgh.ColumnGenerationHeuristicPolicy._run_multi_period_solver
```

````

`````
