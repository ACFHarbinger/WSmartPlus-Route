# {py:mod}`src.policies.route_construction.meta_heuristics.multi_period_particle_swarm_optimization.policy_mp_pso`

```{py:module} src.policies.route_construction.meta_heuristics.multi_period_particle_swarm_optimization.policy_mp_pso
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_particle_swarm_optimization.policy_mp_pso
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MultiPeriodPSOPolicy <src.policies.route_construction.meta_heuristics.multi_period_particle_swarm_optimization.policy_mp_pso.MultiPeriodPSOPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_particle_swarm_optimization.policy_mp_pso.MultiPeriodPSOPolicy
    :summary:
    ```
````

### API

`````{py:class} MultiPeriodPSOPolicy(config: typing.Any = None)
:canonical: src.policies.route_construction.meta_heuristics.multi_period_particle_swarm_optimization.policy_mp_pso.MultiPeriodPSOPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_particle_swarm_optimization.policy_mp_pso.MultiPeriodPSOPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_particle_swarm_optimization.policy_mp_pso.MultiPeriodPSOPolicy.__init__
```

````{py:method} _evaluate(plan: typing.List[typing.List[typing.List[int]]], problem: logic.src.interfaces.context.problem_context.ProblemContext) -> float
:canonical: src.policies.route_construction.meta_heuristics.multi_period_particle_swarm_optimization.policy_mp_pso.MultiPeriodPSOPolicy._evaluate

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_particle_swarm_optimization.policy_mp_pso.MultiPeriodPSOPolicy._evaluate
```

````

````{py:method} _subtract_plans(p1: typing.List[typing.List[typing.List[int]]], p2: typing.List[typing.List[typing.List[int]]]) -> typing.List[tuple]
:canonical: src.policies.route_construction.meta_heuristics.multi_period_particle_swarm_optimization.policy_mp_pso.MultiPeriodPSOPolicy._subtract_plans

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_particle_swarm_optimization.policy_mp_pso.MultiPeriodPSOPolicy._subtract_plans
```

````

````{py:method} _add_velocity(plan: typing.List[typing.List[typing.List[int]]], vel: typing.List[tuple]) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.meta_heuristics.multi_period_particle_swarm_optimization.policy_mp_pso.MultiPeriodPSOPolicy._add_velocity

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_particle_swarm_optimization.policy_mp_pso.MultiPeriodPSOPolicy._add_velocity
```

````

````{py:method} _run_multi_period_solver(problem: logic.src.interfaces.context.problem_context.ProblemContext, multi_day_ctx: typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.meta_heuristics.multi_period_particle_swarm_optimization.policy_mp_pso.MultiPeriodPSOPolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.multi_period_particle_swarm_optimization.policy_mp_pso.MultiPeriodPSOPolicy._run_multi_period_solver
```

````

`````
