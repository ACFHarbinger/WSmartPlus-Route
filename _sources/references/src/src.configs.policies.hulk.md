# {py:mod}`src.configs.policies.hulk`

```{py:module} src.configs.policies.hulk
```

```{autodoc2-docstring} src.configs.policies.hulk
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HULKConfig <src.configs.policies.hulk.HULKConfig>`
  - ```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`register_configs <src.configs.policies.hulk.register_configs>`
  - ```{autodoc2-docstring} src.configs.policies.hulk.register_configs
    :summary:
    ```
````

### API

`````{py:class} HULKConfig
:canonical: src.configs.policies.hulk.HULKConfig

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig
```

````{py:attribute} seed
:canonical: src.configs.policies.hulk.HULKConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.hulk.HULKConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.vrpp
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.hulk.HULKConfig.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.hulk.HULKConfig.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.time_limit
```

````

````{py:attribute} restarts
:canonical: src.configs.policies.hulk.HULKConfig.restarts
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.restarts
```

````

````{py:attribute} restart_threshold
:canonical: src.configs.policies.hulk.HULKConfig.restart_threshold
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.restart_threshold
```

````

````{py:attribute} epsilon
:canonical: src.configs.policies.hulk.HULKConfig.epsilon
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.epsilon
```

````

````{py:attribute} epsilon_decay
:canonical: src.configs.policies.hulk.HULKConfig.epsilon_decay
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.epsilon_decay
```

````

````{py:attribute} min_epsilon
:canonical: src.configs.policies.hulk.HULKConfig.min_epsilon
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.min_epsilon
```

````

````{py:attribute} memory_size
:canonical: src.configs.policies.hulk.HULKConfig.memory_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.memory_size
```

````

````{py:attribute} accept_worse_prob
:canonical: src.configs.policies.hulk.HULKConfig.accept_worse_prob
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.accept_worse_prob
```

````

````{py:attribute} acceptance_decay
:canonical: src.configs.policies.hulk.HULKConfig.acceptance_decay
:type: float
:value: >
   0.99

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.acceptance_decay
```

````

````{py:attribute} start_temp
:canonical: src.configs.policies.hulk.HULKConfig.start_temp
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.start_temp
```

````

````{py:attribute} cooling_rate
:canonical: src.configs.policies.hulk.HULKConfig.cooling_rate
:type: float
:value: >
   0.99

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.cooling_rate
```

````

````{py:attribute} min_temp
:canonical: src.configs.policies.hulk.HULKConfig.min_temp
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.min_temp
```

````

````{py:attribute} min_destroy_size
:canonical: src.configs.policies.hulk.HULKConfig.min_destroy_size
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.min_destroy_size
```

````

````{py:attribute} max_destroy_pct
:canonical: src.configs.policies.hulk.HULKConfig.max_destroy_pct
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.max_destroy_pct
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.hulk.HULKConfig.local_search_iterations
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.local_search_iterations
```

````

````{py:attribute} local_search_operators
:canonical: src.configs.policies.hulk.HULKConfig.local_search_operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.local_search_operators
```

````

````{py:attribute} unstring_operators
:canonical: src.configs.policies.hulk.HULKConfig.unstring_operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.unstring_operators
```

````

````{py:attribute} string_operators
:canonical: src.configs.policies.hulk.HULKConfig.string_operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.string_operators
```

````

````{py:attribute} score_improvement
:canonical: src.configs.policies.hulk.HULKConfig.score_improvement
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.score_improvement
```

````

````{py:attribute} score_accept
:canonical: src.configs.policies.hulk.HULKConfig.score_accept
:type: float
:value: >
   5.0

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.score_accept
```

````

````{py:attribute} score_reject
:canonical: src.configs.policies.hulk.HULKConfig.score_reject
:type: float
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.score_reject
```

````

````{py:attribute} score_best
:canonical: src.configs.policies.hulk.HULKConfig.score_best
:type: float
:value: >
   20.0

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.score_best
```

````

````{py:attribute} weight_learning_rate
:canonical: src.configs.policies.hulk.HULKConfig.weight_learning_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.weight_learning_rate
```

````

````{py:attribute} weight_decay
:canonical: src.configs.policies.hulk.HULKConfig.weight_decay
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.policies.hulk.HULKConfig.weight_decay
```

````

`````

````{py:function} register_configs()
:canonical: src.configs.policies.hulk.register_configs

```{autodoc2-docstring} src.configs.policies.hulk.register_configs
```
````
