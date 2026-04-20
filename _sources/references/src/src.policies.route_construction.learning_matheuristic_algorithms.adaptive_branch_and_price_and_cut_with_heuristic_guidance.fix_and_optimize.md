# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.fix_and_optimize`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.fix_and_optimize
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.fix_and_optimize
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FixAndOptimizeRefiner <src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.fix_and_optimize.FixAndOptimizeRefiner>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.fix_and_optimize.FixAndOptimizeRefiner
    :summary:
    ```
````

### API

`````{py:class} FixAndOptimizeRefiner(tabu_length: int = 10, max_unfix: int = 5)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.fix_and_optimize.FixAndOptimizeRefiner

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.fix_and_optimize.FixAndOptimizeRefiner
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.fix_and_optimize.FixAndOptimizeRefiner.__init__
```

````{py:method} _is_tabu(candidate_set: typing.Set[int]) -> bool
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.fix_and_optimize.FixAndOptimizeRefiner._is_tabu

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.fix_and_optimize.FixAndOptimizeRefiner._is_tabu
```

````

````{py:method} _add_to_tabu(cluster: typing.Set[int])
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.fix_and_optimize.FixAndOptimizeRefiner._add_to_tabu

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.fix_and_optimize.FixAndOptimizeRefiner._add_to_tabu
```

````

````{py:method} select_cluster_overflow_urgency(unrouted: typing.List[int], days_to_overflow: numpy.ndarray) -> typing.Set[int]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.fix_and_optimize.FixAndOptimizeRefiner.select_cluster_overflow_urgency

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.fix_and_optimize.FixAndOptimizeRefiner.select_cluster_overflow_urgency
```

````

````{py:method} select_cluster_scenario_divergence(all_nodes: typing.List[int], scenario_tree: typing.Any, current_day: int) -> typing.Set[int]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.fix_and_optimize.FixAndOptimizeRefiner.select_cluster_scenario_divergence

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.fix_and_optimize.FixAndOptimizeRefiner.select_cluster_scenario_divergence
```

````

````{py:method} refine(current_incumbent: typing.Any, bpc_engine: typing.Any, scenario_tree: typing.Any, current_day: int, days_to_overflow: numpy.ndarray, global_column_pool: typing.List[typing.Any], strategy: str = 'overflow_urgency') -> typing.Any
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.fix_and_optimize.FixAndOptimizeRefiner.refine

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.fix_and_optimize.FixAndOptimizeRefiner.refine
```

````

`````
