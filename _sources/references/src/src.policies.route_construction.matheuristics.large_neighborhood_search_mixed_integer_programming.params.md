# {py:mod}`src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params`

```{py:module} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LNSMIPParams <src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams
    :summary:
    ```
````

### API

`````{py:class} LNSMIPParams
:canonical: src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams

```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams
```

````{py:attribute} k_destroy
:canonical: src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.k_destroy
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.k_destroy
```

````

````{py:attribute} d_destroy
:canonical: src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.d_destroy
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.d_destroy
```

````

````{py:attribute} max_iterations
:canonical: src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.max_iterations
:type: int
:value: >
   200

```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.max_iterations
```

````

````{py:attribute} mip_time_limit
:canonical: src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.mip_time_limit
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.mip_time_limit
```

````

````{py:attribute} mip_gap
:canonical: src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.mip_gap
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.mip_gap
```

````

````{py:attribute} acceptance
:canonical: src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.acceptance
:type: str
:value: >
   'improving'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.acceptance
```

````

````{py:attribute} sa_temperature
:canonical: src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.sa_temperature
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.sa_temperature
```

````

````{py:attribute} sa_cooling
:canonical: src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.sa_cooling
:type: float
:value: >
   0.97

```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.sa_cooling
```

````

````{py:attribute} seed
:canonical: src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.seed
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams
:canonical: src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.matheuristics.large_neighborhood_search_mixed_integer_programming.params.LNSMIPParams.from_config
```

````

`````
