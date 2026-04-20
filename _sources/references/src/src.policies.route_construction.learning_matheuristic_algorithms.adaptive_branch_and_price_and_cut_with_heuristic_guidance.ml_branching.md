# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.ml_branching`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.ml_branching
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.ml_branching
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MLBranchingStrategy <src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.ml_branching.MLBranchingStrategy>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.ml_branching.MLBranchingStrategy
    :summary:
    ```
````

### API

`````{py:class} MLBranchingStrategy(model: typing.Any = None, reliability_c: float = 1.0)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.ml_branching.MLBranchingStrategy

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.ml_branching.MLBranchingStrategy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.ml_branching.MLBranchingStrategy.__init__
```

````{py:method} compute_gnn_features(fractional_vars: typing.List[typing.Any], current_fills: numpy.ndarray, mean_fill_rates: numpy.ndarray, scenario_variances: numpy.ndarray, days_to_overflow: numpy.ndarray) -> typing.Any
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.ml_branching.MLBranchingStrategy.compute_gnn_features

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.ml_branching.MLBranchingStrategy.compute_gnn_features
```

````

````{py:method} _reliability_score(var_id: int) -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.ml_branching.MLBranchingStrategy._reliability_score

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.ml_branching.MLBranchingStrategy._reliability_score
```

````

````{py:method} select_branching_variable(fractional_vars: typing.List[typing.Any], **kwargs) -> typing.Any
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.ml_branching.MLBranchingStrategy.select_branching_variable

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.ml_branching.MLBranchingStrategy.select_branching_variable
```

````

````{py:method} update_pseudocosts(var_id: int, delta_down: float, delta_up: float)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.ml_branching.MLBranchingStrategy.update_pseudocosts

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.ml_branching.MLBranchingStrategy.update_pseudocosts
```

````

`````
