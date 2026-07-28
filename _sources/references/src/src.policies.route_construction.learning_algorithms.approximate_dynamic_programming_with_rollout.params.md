# {py:mod}`src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params`

```{py:module} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params
```

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ADPRolloutParams <src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams
    :summary:
    ```
````

### API

`````{py:class} ADPRolloutParams
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams
```

````{py:attribute} look_ahead_days
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.look_ahead_days
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.look_ahead_days
```

````

````{py:attribute} n_scenarios
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.n_scenarios
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.n_scenarios
```

````

````{py:attribute} fill_threshold
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.fill_threshold
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.fill_threshold
```

````

````{py:attribute} candidate_strategy
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.candidate_strategy
:type: str
:value: >
   'threshold'

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.candidate_strategy
```

````

````{py:attribute} max_candidate_sets
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.max_candidate_sets
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.max_candidate_sets
```

````

````{py:attribute} top_k
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.top_k
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.top_k
```

````

````{py:attribute} stockout_penalty
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.stockout_penalty
:type: float
:value: >
   500.0

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.stockout_penalty
```

````

````{py:attribute} time_limit
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.seed
```

````

````{py:attribute} verbose
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.verbose
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.verbose
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams.from_config
```

````

`````
