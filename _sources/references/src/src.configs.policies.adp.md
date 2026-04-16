# {py:mod}`src.configs.policies.adp`

```{py:module} src.configs.policies.adp
```

```{autodoc2-docstring} src.configs.policies.adp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ADPRolloutConfig <src.configs.policies.adp.ADPRolloutConfig>`
  - ```{autodoc2-docstring} src.configs.policies.adp.ADPRolloutConfig
    :summary:
    ```
````

### API

`````{py:class} ADPRolloutConfig
:canonical: src.configs.policies.adp.ADPRolloutConfig

```{autodoc2-docstring} src.configs.policies.adp.ADPRolloutConfig
```

````{py:attribute} horizon
:canonical: src.configs.policies.adp.ADPRolloutConfig.horizon
:type: int
:value: >
   7

```{autodoc2-docstring} src.configs.policies.adp.ADPRolloutConfig.horizon
```

````

````{py:attribute} look_ahead_days
:canonical: src.configs.policies.adp.ADPRolloutConfig.look_ahead_days
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.adp.ADPRolloutConfig.look_ahead_days
```

````

````{py:attribute} n_scenarios
:canonical: src.configs.policies.adp.ADPRolloutConfig.n_scenarios
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.adp.ADPRolloutConfig.n_scenarios
```

````

````{py:attribute} fill_threshold
:canonical: src.configs.policies.adp.ADPRolloutConfig.fill_threshold
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.adp.ADPRolloutConfig.fill_threshold
```

````

````{py:attribute} candidate_strategy
:canonical: src.configs.policies.adp.ADPRolloutConfig.candidate_strategy
:type: str
:value: >
   'threshold'

```{autodoc2-docstring} src.configs.policies.adp.ADPRolloutConfig.candidate_strategy
```

````

````{py:attribute} max_candidate_sets
:canonical: src.configs.policies.adp.ADPRolloutConfig.max_candidate_sets
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.adp.ADPRolloutConfig.max_candidate_sets
```

````

````{py:attribute} top_k
:canonical: src.configs.policies.adp.ADPRolloutConfig.top_k
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.adp.ADPRolloutConfig.top_k
```

````

````{py:attribute} stockout_penalty
:canonical: src.configs.policies.adp.ADPRolloutConfig.stockout_penalty
:type: float
:value: >
   500.0

```{autodoc2-docstring} src.configs.policies.adp.ADPRolloutConfig.stockout_penalty
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.adp.ADPRolloutConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.adp.ADPRolloutConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.adp.ADPRolloutConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.adp.ADPRolloutConfig.seed
```

````

````{py:attribute} verbose
:canonical: src.configs.policies.adp.ADPRolloutConfig.verbose
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.adp.ADPRolloutConfig.verbose
```

````

`````
