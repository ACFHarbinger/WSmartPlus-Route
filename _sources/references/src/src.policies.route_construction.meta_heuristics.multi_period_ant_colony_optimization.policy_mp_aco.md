# {py:mod}`src.policies.route_construction.meta_heuristics.multi_period_ant_colony_optimization.policy_mp_aco`

```{py:module} src.policies.route_construction.meta_heuristics.multi_period_ant_colony_optimization.policy_mp_aco
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_ant_colony_optimization.policy_mp_aco
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MultiPeriodACOPolicy <src.policies.route_construction.meta_heuristics.multi_period_ant_colony_optimization.policy_mp_aco.MultiPeriodACOPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_ant_colony_optimization.policy_mp_aco.MultiPeriodACOPolicy
    :summary:
    ```
````

### API

`````{py:class} MultiPeriodACOPolicy(config: typing.Any = None)
:canonical: src.policies.route_construction.meta_heuristics.multi_period_ant_colony_optimization.policy_mp_aco.MultiPeriodACOPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_ant_colony_optimization.policy_mp_aco.MultiPeriodACOPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_ant_colony_optimization.policy_mp_aco.MultiPeriodACOPolicy.__init__
```

````{py:method} _evaluate(plan: typing.List[typing.List[typing.List[int]]], problem: logic.src.interfaces.context.problem_context.ProblemContext) -> float
:canonical: src.policies.route_construction.meta_heuristics.multi_period_ant_colony_optimization.policy_mp_aco.MultiPeriodACOPolicy._evaluate

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_ant_colony_optimization.policy_mp_aco.MultiPeriodACOPolicy._evaluate
```

````

````{py:method} _build_ant_solution(problem: logic.src.interfaces.context.problem_context.ProblemContext, pheromones: numpy.ndarray) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.meta_heuristics.multi_period_ant_colony_optimization.policy_mp_aco.MultiPeriodACOPolicy._build_ant_solution

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_ant_colony_optimization.policy_mp_aco.MultiPeriodACOPolicy._build_ant_solution
```

````

````{py:method} _update_pheromones(pheromones: numpy.ndarray, plans: typing.List, best_plan: typing.List, best_prof: float, problem: logic.src.interfaces.context.problem_context.ProblemContext)
:canonical: src.policies.route_construction.meta_heuristics.multi_period_ant_colony_optimization.policy_mp_aco.MultiPeriodACOPolicy._update_pheromones

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_ant_colony_optimization.policy_mp_aco.MultiPeriodACOPolicy._update_pheromones
```

````

````{py:method} _run_multi_period_solver(problem: logic.src.interfaces.context.problem_context.ProblemContext, multi_day_ctx: typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.meta_heuristics.multi_period_ant_colony_optimization.policy_mp_aco.MultiPeriodACOPolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_ant_colony_optimization.policy_mp_aco.MultiPeriodACOPolicy._run_multi_period_solver
```

````

`````
