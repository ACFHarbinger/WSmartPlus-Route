# {py:mod}`src.policies.adaptive_large_neighborhood_search.params`

```{py:module} src.policies.adaptive_large_neighborhood_search.params
```

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ALNSParams <src.policies.adaptive_large_neighborhood_search.params.ALNSParams>`
  - ```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams
    :summary:
    ```
````

### API

`````{py:class} ALNSParams
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams
```

````{py:attribute} time_limit
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.time_limit
```

````

````{py:attribute} max_iterations
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.max_iterations
:type: int
:value: >
   5000

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.max_iterations
```

````

````{py:attribute} start_temp
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.start_temp
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.start_temp
```

````

````{py:attribute} cooling_rate
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.cooling_rate
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.cooling_rate
```

````

````{py:attribute} reaction_factor
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.reaction_factor
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.reaction_factor
```

````

````{py:attribute} min_removal
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.min_removal
:type: int
:value: >
   4

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.min_removal
```

````

````{py:attribute} max_removal_pct
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.max_removal_pct
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.max_removal_pct
```

````

````{py:attribute} segment_size
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.segment_size
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.segment_size
```

````

````{py:attribute} noise_factor
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.noise_factor
:type: float
:value: >
   0.025

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.noise_factor
```

````

````{py:attribute} worst_removal_randomness
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.worst_removal_randomness
:type: float
:value: >
   3.0

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.worst_removal_randomness
```

````

````{py:attribute} shaw_randomization
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.shaw_randomization
:type: float
:value: >
   6.0

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.shaw_randomization
```

````

````{py:attribute} max_removal_cap
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.max_removal_cap
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.max_removal_cap
```

````

````{py:attribute} regret_pool
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.regret_pool
:type: str
:value: >
   'regret234'

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.regret_pool
```

````

````{py:attribute} sigma_1
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.sigma_1
:type: float
:value: >
   33.0

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.sigma_1
```

````

````{py:attribute} sigma_2
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.sigma_2
:type: float
:value: >
   9.0

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.sigma_2
```

````

````{py:attribute} sigma_3
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.sigma_3
:type: float
:value: >
   13.0

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.sigma_3
```

````

````{py:attribute} vrpp
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.profit_aware_operators
```

````

````{py:attribute} extended_operators
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.extended_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.extended_operators
```

````

````{py:attribute} seed
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.seed
```

````

````{py:attribute} engine
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.engine
:type: str
:value: >
   'custom'

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.engine
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.adaptive_large_neighborhood_search.params.ALNSParams
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.from_config
```

````

`````
