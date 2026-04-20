# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TemporalBendersCoordinator <src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.logger>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.logger
```

````

`````{py:class} TemporalBendersCoordinator(tree: typing.Any, prize_engine: typing.Any, capacity: float, revenue: float, cost_unit: float)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator.__init__
```

````{py:method} solve(**kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator.solve

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator.solve
```

````

````{py:method} generate_benders_cut(day: int, scenario_id: int, z_bar: typing.Dict[int, int], subproblem_profit: float, subproblem_duals: typing.Dict[int, float], scenario_prob: float) -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator.generate_benders_cut

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator.generate_benders_cut
```

````

`````
