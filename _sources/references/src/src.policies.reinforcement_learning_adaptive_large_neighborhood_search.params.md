# {py:mod}`src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params`

```{py:module} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params
```

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RLALNSParams <src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams>`
  - ```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams
    :summary:
    ```
````

### API

`````{py:class} RLALNSParams
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams
```

````{py:attribute} time_limit
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.time_limit
```

````

````{py:attribute} max_iterations
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.max_iterations
:type: int
:value: >
   5000

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.max_iterations
```

````

````{py:attribute} start_temp
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.start_temp
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.start_temp
```

````

````{py:attribute} cooling_rate
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.cooling_rate
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.cooling_rate
```

````

````{py:attribute} min_removal
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.min_removal
:type: int
:value: >
   1

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.min_removal
```

````

````{py:attribute} max_removal_pct
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.max_removal_pct
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.max_removal_pct
```

````

````{py:attribute} rl_config
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.rl_config
:type: logic.src.configs.policies.other.RLConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.rl_config
```

````

````{py:property} rl_algorithm
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.rl_algorithm
:type: str

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.rl_algorithm
```

````

````{py:property} alpha
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.alpha
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.alpha
```

````

````{py:property} gamma
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.gamma
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.gamma
```

````

````{py:property} epsilon
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.epsilon
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.epsilon
```

````

````{py:property} ucb_c
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.ucb_c
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.ucb_c
```

````

````{py:property} reward_new_global_best
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.reward_new_global_best
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.reward_new_global_best
```

````

````{py:property} reward_improved_current
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.reward_improved_current
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.reward_improved_current
```

````

````{py:property} reward_accepted_worse
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.reward_accepted_worse
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.reward_accepted_worse
```

````

````{py:property} reward_rejected
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.reward_rejected
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.reward_rejected
```

````

````{py:property} adaptive_rewards
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.adaptive_rewards
:type: bool

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.adaptive_rewards
```

````

````{py:property} normalize_rewards
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.normalize_rewards
:type: bool

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.normalize_rewards
```

````

````{py:property} epsilon_decay
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.epsilon_decay
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.epsilon_decay
```

````

````{py:property} epsilon_min
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.epsilon_min
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.epsilon_min
```

````

````{py:property} ucb_gamma
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.ucb_gamma
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.ucb_gamma
```

````

````{py:property} ucb_window_size
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.ucb_window_size
:type: int

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.ucb_window_size
```

````

````{py:property} ts_alpha_prior
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.ts_alpha_prior
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.ts_alpha_prior
```

````

````{py:property} ts_beta_prior
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.ts_beta_prior
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.ts_beta_prior
```

````

````{py:property} exp3_gamma
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.exp3_gamma
:type: float

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.exp3_gamma
```

````

````{py:property} progress_thresholds
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.progress_thresholds
:type: list[float]

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.progress_thresholds
```

````

````{py:property} stagnation_thresholds
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.stagnation_thresholds
:type: list[int]

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.stagnation_thresholds
```

````

````{py:property} diversity_thresholds
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.diversity_thresholds
:type: list[float]

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.diversity_thresholds
```

````

````{py:method} from_config(config: logic.src.configs.policies.RLALNSConfig) -> src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams
:canonical: src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams.from_config
```

````

`````
