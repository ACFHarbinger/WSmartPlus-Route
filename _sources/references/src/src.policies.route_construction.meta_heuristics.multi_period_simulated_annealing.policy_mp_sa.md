# {py:mod}`src.policies.route_construction.meta_heuristics.multi_period_simulated_annealing.policy_mp_sa`

```{py:module} src.policies.route_construction.meta_heuristics.multi_period_simulated_annealing.policy_mp_sa
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_simulated_annealing.policy_mp_sa
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MultiPeriodSimulatedAnnealingPolicy <src.policies.route_construction.meta_heuristics.multi_period_simulated_annealing.policy_mp_sa.MultiPeriodSimulatedAnnealingPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_simulated_annealing.policy_mp_sa.MultiPeriodSimulatedAnnealingPolicy
    :summary:
    ```
````

### API

`````{py:class} MultiPeriodSimulatedAnnealingPolicy(config: typing.Any = None)
:canonical: src.policies.route_construction.meta_heuristics.multi_period_simulated_annealing.policy_mp_sa.MultiPeriodSimulatedAnnealingPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_simulated_annealing.policy_mp_sa.MultiPeriodSimulatedAnnealingPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_simulated_annealing.policy_mp_sa.MultiPeriodSimulatedAnnealingPolicy.__init__
```

````{py:method} _evaluate(plan: typing.List[typing.List[typing.List[int]]], problem: logic.src.interfaces.context.problem_context.ProblemContext) -> float
:canonical: src.policies.route_construction.meta_heuristics.multi_period_simulated_annealing.policy_mp_sa.MultiPeriodSimulatedAnnealingPolicy._evaluate

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_simulated_annealing.policy_mp_sa.MultiPeriodSimulatedAnnealingPolicy._evaluate
```

````

````{py:method} _neighbor(plan: typing.List[typing.List[typing.List[int]]], problem: logic.src.interfaces.context.problem_context.ProblemContext) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.meta_heuristics.multi_period_simulated_annealing.policy_mp_sa.MultiPeriodSimulatedAnnealingPolicy._neighbor

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_simulated_annealing.policy_mp_sa.MultiPeriodSimulatedAnnealingPolicy._neighbor
```

````

````{py:method} _run_multi_period_solver(problem: logic.src.interfaces.context.problem_context.ProblemContext, multi_day_ctx: typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.meta_heuristics.multi_period_simulated_annealing.policy_mp_sa.MultiPeriodSimulatedAnnealingPolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_simulated_annealing.policy_mp_sa.MultiPeriodSimulatedAnnealingPolicy._run_multi_period_solver
```

````

`````
