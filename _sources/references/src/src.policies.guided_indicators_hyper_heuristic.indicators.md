# {py:mod}`src.policies.guided_indicators_hyper_heuristic.indicators`

```{py:module} src.policies.guided_indicators_hyper_heuristic.indicators
```

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImprovementRateIndicator <src.policies.guided_indicators_hyper_heuristic.indicators.ImprovementRateIndicator>`
  - ```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.ImprovementRateIndicator
    :summary:
    ```
* - {py:obj}`TimeBasedIndicator <src.policies.guided_indicators_hyper_heuristic.indicators.TimeBasedIndicator>`
  - ```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.TimeBasedIndicator
    :summary:
    ```
````

### API

`````{py:class} ImprovementRateIndicator(window_size: int = 20)
:canonical: src.policies.guided_indicators_hyper_heuristic.indicators.ImprovementRateIndicator

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.ImprovementRateIndicator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.ImprovementRateIndicator.__init__
```

````{py:method} update(operator: str, improvement: float) -> None
:canonical: src.policies.guided_indicators_hyper_heuristic.indicators.ImprovementRateIndicator.update

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.ImprovementRateIndicator.update
```

````

````{py:method} get_score(operator: str, operator_improvements: typing.Deque[float]) -> float
:canonical: src.policies.guided_indicators_hyper_heuristic.indicators.ImprovementRateIndicator.get_score

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.ImprovementRateIndicator.get_score
```

````

`````

`````{py:class} TimeBasedIndicator(window_size: int = 20)
:canonical: src.policies.guided_indicators_hyper_heuristic.indicators.TimeBasedIndicator

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.TimeBasedIndicator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.TimeBasedIndicator.__init__
```

````{py:method} update(operator: str, elapsed_time: float) -> None
:canonical: src.policies.guided_indicators_hyper_heuristic.indicators.TimeBasedIndicator.update

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.TimeBasedIndicator.update
```

````

````{py:method} get_score(operator: str, operator_times: typing.Deque[float]) -> float
:canonical: src.policies.guided_indicators_hyper_heuristic.indicators.TimeBasedIndicator.get_score

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.TimeBasedIndicator.get_score
```

````

`````
