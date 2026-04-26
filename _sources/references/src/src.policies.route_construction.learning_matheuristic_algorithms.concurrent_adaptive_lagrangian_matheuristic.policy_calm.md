# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_RunState <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState
    :summary:
    ```
* - {py:obj}`CALMPolicy <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy
    :summary:
    ```
````

### API

`````{py:class} _RunState
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState
```

````{py:attribute} tables
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.tables
:type: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadTables
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.tables
```

````

````{py:attribute} scenario_tree
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.scenario_tree
:type: logic.src.pipeline.simulations.bins.prediction.ScenarioTree
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.scenario_tree
```

````

````{py:attribute} lag_state
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.lag_state
:type: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianState
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.lag_state
```

````

````{py:attribute} coordinator
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.coordinator
:type: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.coordinator
```

````

````{py:attribute} oracle
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.oracle
:type: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.oracle.InsertionCostOracle
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.oracle
```

````

````{py:attribute} bandit
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.bandit
:type: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.bandit.LinUCBBandit
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.bandit
```

````

````{py:attribute} regret_preprocessor
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.regret_preprocessor
:type: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.regret.RegretPreprocessor
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.regret_preprocessor
```

````

````{py:attribute} best_primal
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.best_primal
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.best_primal
```

````

````{py:attribute} best_per_period_tours
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.best_per_period_tours
:type: typing.Dict[int, typing.List[int]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.best_per_period_tours
```

````

````{py:attribute} best_per_period_selection
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.best_per_period_selection
:type: typing.Dict[int, typing.List[int]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.best_per_period_selection
```

````

````{py:attribute} best_per_period_cost
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.best_per_period_cost
:type: typing.Dict[int, float]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.best_per_period_cost
```

````

````{py:attribute} outer_iter
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.outer_iter
:type: int
:value: >
   0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.outer_iter
```

````

````{py:attribute} iters_since_improvement
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.iters_since_improvement
:type: int
:value: >
   0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.iters_since_improvement
```

````

````{py:attribute} prior_cuts
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.prior_cuts
:type: typing.Dict[int, typing.Dict[int, float]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.prior_cuts
```

````

````{py:attribute} selection_history
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.selection_history
:type: typing.List[typing.List[src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.selection_history
```

````

````{py:attribute} start_time
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.start_time
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState.start_time
```

````

`````

`````{py:class} CALMPolicy(config: typing.Any = None)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy.__init__
```

````{py:method} _config_class()
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._get_config_key
```

````

````{py:method} _run_multi_period_solver(problem: logic.src.interfaces.context.problem_context.ProblemContext, multi_day_ctx: typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._run_multi_period_solver
```

````

````{py:method} _initialise(problem: logic.src.interfaces.context.problem_context.ProblemContext) -> src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._initialise

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._initialise
```

````

````{py:method} _solve_selection_layer(*, state: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState, problem: logic.src.interfaces.context.problem_context.ProblemContext, engine: str, plan: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.regret.RegretPlan) -> typing.List[src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._solve_selection_layer

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._solve_selection_layer
```

````

````{py:method} _solve_routing_layer(*, state: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState, problem: logic.src.interfaces.context.problem_context.ProblemContext, selection_results: typing.List[src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult]) -> typing.List[src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.routing.RoutingResult]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._solve_routing_layer

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._solve_routing_layer
```

````

````{py:method} _generate_cuts(*, strategy: str, selection_results: typing.List[src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult], state: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState, problem: logic.src.interfaces.context.problem_context.ProblemContext) -> typing.Dict[int, typing.Dict[int, float]]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._generate_cuts

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._generate_cuts
```

````

````{py:method} _assemble_primal(state: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState, problem: logic.src.interfaces.context.problem_context.ProblemContext) -> typing.Tuple[float, bool]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._assemble_primal

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._assemble_primal
```

````

````{py:method} _stockout_penalty_estimate(state: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState, problem: logic.src.interfaces.context.problem_context.ProblemContext) -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._stockout_penalty_estimate

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._stockout_penalty_estimate
```

````

````{py:method} _should_stop(state: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState) -> bool
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._should_stop

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._should_stop
```

````

````{py:method} _assemble_x_K(selection_results: typing.List[src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult], n_bins: int, horizon: int) -> numpy.ndarray
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._assemble_x_K

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._assemble_x_K
```

````

````{py:method} _compute_full_lagrangian(selection_results: typing.List[src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult], routing_results: typing.List[src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.routing.RoutingResult]) -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._compute_full_lagrangian

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._compute_full_lagrangian
```

````

````{py:method} _build_context(state: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState, problem: logic.src.interfaces.context.problem_context.ProblemContext) -> numpy.ndarray
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._build_context

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._build_context
```

````

````{py:method} _log_iter(state: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState, arm_idx: int, engine: str, cut_strategy: str, primal: float, lagrangian: float, coord_stats: typing.Dict[str, float]) -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._log_iter

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._log_iter
```

````

````{py:method} _package_solution(state: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm._RunState, problem: logic.src.interfaces.context.problem_context.ProblemContext) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._package_solution

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.policy_calm.CALMPolicy._package_solution
```

````

`````
