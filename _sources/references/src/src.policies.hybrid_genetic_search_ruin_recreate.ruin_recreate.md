# {py:mod}`src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate`

```{py:module} src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate
```

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AdaptiveOperatorManager <src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.AdaptiveOperatorManager>`
  - ```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.AdaptiveOperatorManager
    :summary:
    ```
* - {py:obj}`RuinRecreateOperator <src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.RuinRecreateOperator>`
  - ```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.RuinRecreateOperator
    :summary:
    ```
````

### API

`````{py:class} AdaptiveOperatorManager(destroy_operators: typing.List[str], repair_operators: typing.List[str], reaction_factor: float = 0.1, decay_parameter: float = 0.95)
:canonical: src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.AdaptiveOperatorManager

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.AdaptiveOperatorManager
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.AdaptiveOperatorManager.__init__
```

````{py:method} select_operators(rng: random.Random) -> typing.Tuple[str, str]
:canonical: src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.AdaptiveOperatorManager.select_operators

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.AdaptiveOperatorManager.select_operators
```

````

````{py:method} update_scores(destroy_op: str, repair_op: str, score: float) -> None
:canonical: src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.AdaptiveOperatorManager.update_scores

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.AdaptiveOperatorManager.update_scores
```

````

````{py:method} decay_weights() -> None
:canonical: src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.AdaptiveOperatorManager.decay_weights

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.AdaptiveOperatorManager.decay_weights
```

````

````{py:method} entropy() -> float
:canonical: src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.AdaptiveOperatorManager.entropy

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.AdaptiveOperatorManager.entropy
```

````

````{py:method} _compute_entropy(weights: typing.List[float]) -> float
:canonical: src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.AdaptiveOperatorManager._compute_entropy
:staticmethod:

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.AdaptiveOperatorManager._compute_entropy
```

````

````{py:method} _roulette_wheel(weights: typing.Dict[str, float], rng: random.Random) -> str
:canonical: src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.AdaptiveOperatorManager._roulette_wheel
:staticmethod:

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.AdaptiveOperatorManager._roulette_wheel
```

````

`````

`````{py:class} RuinRecreateOperator(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, params: src.policies.hybrid_genetic_search_ruin_recreate.params.HGSRRParams, split_manager, seed: typing.Optional[int] = None)
:canonical: src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.RuinRecreateOperator

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.RuinRecreateOperator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.RuinRecreateOperator.__init__
```

````{py:method} apply(individual: logic.src.policies.hybrid_genetic_search.Individual, destroy_operator: str, repair_operator: str) -> logic.src.policies.hybrid_genetic_search.Individual
:canonical: src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.RuinRecreateOperator.apply

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.RuinRecreateOperator.apply
```

````

````{py:method} _apply_destroy(routes: typing.List[typing.List[int]], operator_name: str, n_remove: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.RuinRecreateOperator._apply_destroy

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.RuinRecreateOperator._apply_destroy
```

````

````{py:method} _apply_repair(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], operator_name: str) -> typing.List[typing.List[int]]
:canonical: src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.RuinRecreateOperator._apply_repair

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.RuinRecreateOperator._apply_repair
```

````

````{py:method} _routes_to_giant_tour(routes: typing.List[typing.List[int]]) -> typing.List[int]
:canonical: src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.RuinRecreateOperator._routes_to_giant_tour
:staticmethod:

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_recreate.ruin_recreate.RuinRecreateOperator._routes_to_giant_tour
```

````

`````
