# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ABPCHGPolicy <src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy
    :summary:
    ```
````

### API

`````{py:class} ABPCHGPolicy(config: typing.Optional[logic.src.configs.policies.abpc_hg.ABPCHGConfig] = None)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy.__init__
```

````{py:method} _config_class() -> typing.Type[logic.src.configs.policies.abpc_hg.ABPCHGConfig]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._get_config_key
```

````

````{py:method} _build_prize_engine(tree: typing.Any, capacity: float) -> src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_prize_engine.ScenarioPrizeEngine
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._build_prize_engine

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._build_prize_engine
```

````

````{py:method} _build_ph_loop(num_scenarios: int) -> src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.progressive_hedging_cg.ProgressiveHedgingCGLoop
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._build_ph_loop

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._build_ph_loop
```

````

````{py:method} _build_alns_pricer(exact_pricer: logic.src.policies.helpers.solvers_and_matheuristics.RCSPPSolver) -> src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.alns_pricer.ALNSMultiPeriodPricer
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._build_alns_pricer

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._build_alns_pricer
```

````

````{py:method} _build_dive_heuristic() -> src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.dive_and_price.DiveAndPricePrimalHeuristic
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._build_dive_heuristic

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._build_dive_heuristic
```

````

````{py:method} _build_fix_optimizer() -> src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.fix_and_optimize.FixAndOptimizeRefiner
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._build_fix_optimizer

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._build_fix_optimizer
```

````

````{py:method} _build_ml_branching() -> src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.ml_branching.MLBranchingStrategy
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._build_ml_branching

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._build_ml_branching
```

````

````{py:method} _build_scenario_branching() -> src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_branching.ScenarioConsistentBranching
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._build_scenario_branching

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._build_scenario_branching
```

````

````{py:method} _build_coordinator(tree: typing.Any, prize_engine: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_prize_engine.ScenarioPrizeEngine, capacity: float, revenue: float, cost_unit: float, num_scenarios: int, exact_pricer: typing.Optional[logic.src.policies.helpers.solvers_and_matheuristics.RCSPPSolver] = None) -> src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._build_coordinator

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._build_coordinator
```

````

````{py:method} _run_multi_period_solver(problem: logic.src.interfaces.context.problem_context.ProblemContext, multi_day_ctx: typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg.ABPCHGPolicy._run_multi_period_solver
```

````

`````
