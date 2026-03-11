# {py:mod}`src.policies.guided_indicators_hyper_heuristic.params`

```{py:module} src.policies.guided_indicators_hyper_heuristic.params
```

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GIHHParams <src.policies.guided_indicators_hyper_heuristic.params.GIHHParams>`
  - ```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams
    :summary:
    ```
````

### API

`````{py:class} GIHHParams
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams
```

````{py:attribute} time_limit
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.time_limit
```

````

````{py:attribute} max_iterations
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.max_iterations
```

````

````{py:attribute} seed
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.seed
```

````

````{py:attribute} move_operators
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.move_operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.move_operators
```

````

````{py:attribute} perturbation_operators
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.perturbation_operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.perturbation_operators
```

````

````{py:attribute} iri_weight
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.iri_weight
:type: float
:value: >
   0.6

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.iri_weight
```

````

````{py:attribute} tbi_weight
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.tbi_weight
:type: float
:value: >
   0.4

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.tbi_weight
```

````

````{py:attribute} learning_rate
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.learning_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.learning_rate
```

````

````{py:attribute} memory_size
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.memory_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.memory_size
```

````

````{py:attribute} epsilon
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.epsilon
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.epsilon
```

````

````{py:attribute} epsilon_decay
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.epsilon_decay
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.epsilon_decay
```

````

````{py:attribute} min_epsilon
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.min_epsilon
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.min_epsilon
```

````

````{py:attribute} accept_equal
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.accept_equal
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.accept_equal
```

````

````{py:attribute} accept_worse_prob
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.accept_worse_prob
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.accept_worse_prob
```

````

````{py:attribute} acceptance_decay
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.acceptance_decay
:type: float
:value: >
   0.99

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.acceptance_decay
```

````

````{py:attribute} iri_window
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.iri_window
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.iri_window
```

````

````{py:attribute} tbi_window
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.tbi_window
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.tbi_window
```

````

````{py:attribute} restarts
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.restarts
:type: int
:value: >
   1

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.restarts
```

````

````{py:attribute} restart_threshold
:canonical: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.restart_threshold
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.params.GIHHParams.restart_threshold
```

````

`````
