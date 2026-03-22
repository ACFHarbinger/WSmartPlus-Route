# {py:mod}`src.policies.hyper_heuristic_us_lk.adaptive_selection`

```{py:module} src.policies.hyper_heuristic_us_lk.adaptive_selection
```

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.adaptive_selection
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AdaptiveOperatorSelector <src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector>`
  - ```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector
    :summary:
    ```
````

### API

`````{py:class} AdaptiveOperatorSelector(operators: typing.List[str], epsilon: float = 0.3, memory_size: int = 50, learning_rate: float = 0.1, weight_decay: float = 0.95, seed: typing.Optional[int] = 42)
:canonical: src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector.__init__
```

````{py:method} select_operator() -> str
:canonical: src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector.select_operator

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector.select_operator
```

````

````{py:method} _select_by_weight() -> str
:canonical: src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector._select_by_weight

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector._select_by_weight
```

````

````{py:method} update(operator: str, improvement: float, elapsed_time: float, is_best: bool = False)
:canonical: src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector.update

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector.update
```

````

````{py:method} _calculate_score(improvement: float, is_best: bool) -> float
:canonical: src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector._calculate_score

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector._calculate_score
```

````

````{py:method} _update_weight(operator: str)
:canonical: src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector._update_weight

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector._update_weight
```

````

````{py:method} decay_epsilon(decay_rate: float = 0.995, min_epsilon: float = 0.05)
:canonical: src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector.decay_epsilon

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector.decay_epsilon
```

````

````{py:method} get_statistics() -> typing.Dict[str, typing.Dict]
:canonical: src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector.get_statistics

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector.get_statistics
```

````

````{py:method} reset_statistics()
:canonical: src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector.reset_statistics

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.adaptive_selection.AdaptiveOperatorSelector.reset_statistics
```

````

`````
