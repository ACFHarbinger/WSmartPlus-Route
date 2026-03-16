# {py:mod}`src.policies.hyper_heuristic_us_lk.params`

```{py:module} src.policies.hyper_heuristic_us_lk.params
```

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HULKParams <src.policies.hyper_heuristic_us_lk.params.HULKParams>`
  - ```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams
    :summary:
    ```
````

### API

`````{py:class} HULKParams
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams
```

````{py:attribute} max_iterations
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.time_limit
```

````

````{py:attribute} restarts
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.restarts
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.restarts
```

````

````{py:attribute} restart_threshold
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.restart_threshold
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.restart_threshold
```

````

````{py:attribute} epsilon
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.epsilon
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.epsilon
```

````

````{py:attribute} epsilon_decay
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.epsilon_decay
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.epsilon_decay
```

````

````{py:attribute} min_epsilon
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.min_epsilon
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.min_epsilon
```

````

````{py:attribute} memory_size
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.memory_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.memory_size
```

````

````{py:attribute} accept_worse_prob
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.accept_worse_prob
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.accept_worse_prob
```

````

````{py:attribute} acceptance_decay
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.acceptance_decay
:type: float
:value: >
   0.99

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.acceptance_decay
```

````

````{py:attribute} start_temp
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.start_temp
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.start_temp
```

````

````{py:attribute} cooling_rate
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.cooling_rate
:type: float
:value: >
   0.99

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.cooling_rate
```

````

````{py:attribute} min_temp
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.min_temp
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.min_temp
```

````

````{py:attribute} min_destroy_size
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.min_destroy_size
:type: int
:value: >
   2

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.min_destroy_size
```

````

````{py:attribute} max_destroy_pct
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.max_destroy_pct
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.max_destroy_pct
```

````

````{py:attribute} local_search_iterations
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.local_search_iterations
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.local_search_iterations
```

````

````{py:attribute} local_search_operators
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.local_search_operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.local_search_operators
```

````

````{py:attribute} unstring_operators
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.unstring_operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.unstring_operators
```

````

````{py:attribute} string_operators
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.string_operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.string_operators
```

````

````{py:attribute} operator_weights
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.operator_weights
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.operator_weights
```

````

````{py:attribute} score_improvement
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.score_improvement
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.score_improvement
```

````

````{py:attribute} score_accept
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.score_accept
:type: float
:value: >
   5.0

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.score_accept
```

````

````{py:attribute} score_reject
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.score_reject
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.score_reject
```

````

````{py:attribute} score_best
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.score_best
:type: float
:value: >
   20.0

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.score_best
```

````

````{py:attribute} weight_learning_rate
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.weight_learning_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.weight_learning_rate
```

````

````{py:attribute} weight_decay
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.weight_decay
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.weight_decay
```

````

````{py:method} __post_init__()
:canonical: src.policies.hyper_heuristic_us_lk.params.HULKParams.__post_init__

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.params.HULKParams.__post_init__
```

````

`````
