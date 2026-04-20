# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.dive_and_price`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.dive_and_price
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.dive_and_price
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DiveAndPricePrimalHeuristic <src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.dive_and_price.DiveAndPricePrimalHeuristic>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.dive_and_price.DiveAndPricePrimalHeuristic
    :summary:
    ```
````

### API

`````{py:class} DiveAndPricePrimalHeuristic(penalty_M: float = 10000.0)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.dive_and_price.DiveAndPricePrimalHeuristic

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.dive_and_price.DiveAndPricePrimalHeuristic
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.dive_and_price.DiveAndPricePrimalHeuristic.__init__
```

````{py:method} evaluate_scenario_consensus(route_nodes: typing.List[int], scenario_tree: typing.Any) -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.dive_and_price.DiveAndPricePrimalHeuristic.evaluate_scenario_consensus

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.dive_and_price.DiveAndPricePrimalHeuristic.evaluate_scenario_consensus
```

````

````{py:method} select_column_to_fix(fractional_columns: typing.List[typing.Tuple[int, float, typing.List[int]]], scenario_tree: typing.Any) -> int
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.dive_and_price.DiveAndPricePrimalHeuristic.select_column_to_fix

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.dive_and_price.DiveAndPricePrimalHeuristic.select_column_to_fix
```

````

````{py:method} execute(rmp: typing.Any, fractional_columns: typing.List[typing.Tuple[int, float, typing.List[int]]], scenario_tree: typing.Any) -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.dive_and_price.DiveAndPricePrimalHeuristic.execute

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.dive_and_price.DiveAndPricePrimalHeuristic.execute
```

````

`````
