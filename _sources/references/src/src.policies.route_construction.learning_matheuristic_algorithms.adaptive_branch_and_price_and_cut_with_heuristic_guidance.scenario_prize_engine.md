# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_prize_engine`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_prize_engine
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_prize_engine
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ScenarioPrizeEngine <src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_prize_engine.ScenarioPrizeEngine>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_prize_engine.ScenarioPrizeEngine
    :summary:
    ```
````

### API

`````{py:class} ScenarioPrizeEngine(scenario_tree: logic.src.pipeline.simulations.bins.prediction.ScenarioTree, gamma: float = 0.95, tau: float = 100.0, overflow_penalty: float = 2.0)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_prize_engine.ScenarioPrizeEngine

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_prize_engine.ScenarioPrizeEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_prize_engine.ScenarioPrizeEngine.__init__
```

````{py:method} compute_prizes(current_wastes: numpy.ndarray, bin_stats: typing.Dict[str, numpy.ndarray], revenue: float, days_remaining: int) -> typing.Dict[int, float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_prize_engine.ScenarioPrizeEngine.compute_prizes

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_prize_engine.ScenarioPrizeEngine.compute_prizes
```

````

````{py:method} _expected_future_value_leave(idx: int, current_fill: float, mean_rate: float, variance: float, days_to_overflow: float, days_remaining: int, revenue: float) -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_prize_engine.ScenarioPrizeEngine._expected_future_value_leave

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_prize_engine.ScenarioPrizeEngine._expected_future_value_leave
```

````

````{py:method} _expected_future_value_visit(idx: int, mean_rate: float, variance: float, days_remaining: int, revenue: float) -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_prize_engine.ScenarioPrizeEngine._expected_future_value_visit

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_prize_engine.ScenarioPrizeEngine._expected_future_value_visit
```

````

````{py:method} scenario_weighted_prizes(day: int, revenue: float, days_remaining: int) -> typing.Dict[int, float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_prize_engine.ScenarioPrizeEngine.scenario_weighted_prizes

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.scenario_prize_engine.ScenarioPrizeEngine.scenario_weighted_prizes
```

````

`````
