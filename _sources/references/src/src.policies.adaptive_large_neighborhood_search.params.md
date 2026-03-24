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
   100.0

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
   1

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

````{py:attribute} seed
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.seed
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.adaptive_large_neighborhood_search.params.ALNSParams
:canonical: src.policies.adaptive_large_neighborhood_search.params.ALNSParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.params.ALNSParams.from_config
```

````

`````
