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

* - {py:obj}`ScoreAIndicator <src.policies.guided_indicators_hyper_heuristic.indicators.ScoreAIndicator>`
  - ```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.ScoreAIndicator
    :summary:
    ```
* - {py:obj}`ScoreBIndicator <src.policies.guided_indicators_hyper_heuristic.indicators.ScoreBIndicator>`
  - ```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.ScoreBIndicator
    :summary:
    ```
````

### API

`````{py:class} ScoreAIndicator()
:canonical: src.policies.guided_indicators_hyper_heuristic.indicators.ScoreAIndicator

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.ScoreAIndicator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.ScoreAIndicator.__init__
```

````{py:method} update(operator: str, accepted: bool) -> None
:canonical: src.policies.guided_indicators_hyper_heuristic.indicators.ScoreAIndicator.update

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.ScoreAIndicator.update
```

````

````{py:method} get_score(operator: str) -> float
:canonical: src.policies.guided_indicators_hyper_heuristic.indicators.ScoreAIndicator.get_score

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.ScoreAIndicator.get_score
```

````

````{py:method} reset(operator: str) -> None
:canonical: src.policies.guided_indicators_hyper_heuristic.indicators.ScoreAIndicator.reset

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.ScoreAIndicator.reset
```

````

`````

`````{py:class} ScoreBIndicator()
:canonical: src.policies.guided_indicators_hyper_heuristic.indicators.ScoreBIndicator

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.ScoreBIndicator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.ScoreBIndicator.__init__
```

````{py:method} update(operator: str, revenue_improved: bool, cost_improved: bool) -> None
:canonical: src.policies.guided_indicators_hyper_heuristic.indicators.ScoreBIndicator.update

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.ScoreBIndicator.update
```

````

````{py:method} get_score(operator: str) -> float
:canonical: src.policies.guided_indicators_hyper_heuristic.indicators.ScoreBIndicator.get_score

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.ScoreBIndicator.get_score
```

````

````{py:method} reset(operator: str) -> None
:canonical: src.policies.guided_indicators_hyper_heuristic.indicators.ScoreBIndicator.reset

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.indicators.ScoreBIndicator.reset
```

````

`````
