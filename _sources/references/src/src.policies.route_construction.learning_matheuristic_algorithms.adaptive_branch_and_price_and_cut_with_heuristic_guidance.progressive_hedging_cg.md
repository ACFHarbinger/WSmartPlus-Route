# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.progressive_hedging_cg`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.progressive_hedging_cg
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.progressive_hedging_cg
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ProgressiveHedgingCGLoop <src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.progressive_hedging_cg.ProgressiveHedgingCGLoop>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.progressive_hedging_cg.ProgressiveHedgingCGLoop
    :summary:
    ```
````

### API

`````{py:class} ProgressiveHedgingCGLoop(num_scenarios: int, base_rho: float = 1.0)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.progressive_hedging_cg.ProgressiveHedgingCGLoop

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.progressive_hedging_cg.ProgressiveHedgingCGLoop
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.progressive_hedging_cg.ProgressiveHedgingCGLoop.__init__
```

````{py:method} update_x_bar(scenario_solutions: typing.Dict[int, typing.Dict[int, float]])
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.progressive_hedging_cg.ProgressiveHedgingCGLoop.update_x_bar

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.progressive_hedging_cg.ProgressiveHedgingCGLoop.update_x_bar
```

````

````{py:method} compute_dynamic_penalty(var_id: int, scenario_prizes: typing.Dict[int, float]) -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.progressive_hedging_cg.ProgressiveHedgingCGLoop.compute_dynamic_penalty

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.progressive_hedging_cg.ProgressiveHedgingCGLoop.compute_dynamic_penalty
```

````

````{py:method} calculate_augmented_reduced_cost(route_nodes: typing.List[int], dist_matrix: numpy.ndarray, scenario_prizes: typing.Dict[int, float], dual_values: typing.Dict[int, float], scenario_id: int, route_id: int, current_x_k: float) -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.progressive_hedging_cg.ProgressiveHedgingCGLoop.calculate_augmented_reduced_cost

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.progressive_hedging_cg.ProgressiveHedgingCGLoop.calculate_augmented_reduced_cost
```

````

`````
