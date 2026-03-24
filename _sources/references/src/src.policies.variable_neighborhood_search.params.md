# {py:mod}`src.policies.variable_neighborhood_search.params`

```{py:module} src.policies.variable_neighborhood_search.params
```

```{autodoc2-docstring} src.policies.variable_neighborhood_search.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VNSParams <src.policies.variable_neighborhood_search.params.VNSParams>`
  - ```{autodoc2-docstring} src.policies.variable_neighborhood_search.params.VNSParams
    :summary:
    ```
````

### API

`````{py:class} VNSParams
:canonical: src.policies.variable_neighborhood_search.params.VNSParams

```{autodoc2-docstring} src.policies.variable_neighborhood_search.params.VNSParams
```

````{py:attribute} k_max
:canonical: src.policies.variable_neighborhood_search.params.VNSParams.k_max
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.variable_neighborhood_search.params.VNSParams.k_max
```

````

````{py:attribute} max_iterations
:canonical: src.policies.variable_neighborhood_search.params.VNSParams.max_iterations
:type: int
:value: >
   200

```{autodoc2-docstring} src.policies.variable_neighborhood_search.params.VNSParams.max_iterations
```

````

````{py:attribute} local_search_iterations
:canonical: src.policies.variable_neighborhood_search.params.VNSParams.local_search_iterations
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.variable_neighborhood_search.params.VNSParams.local_search_iterations
```

````

````{py:attribute} n_removal
:canonical: src.policies.variable_neighborhood_search.params.VNSParams.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.policies.variable_neighborhood_search.params.VNSParams.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.policies.variable_neighborhood_search.params.VNSParams.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.variable_neighborhood_search.params.VNSParams.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.policies.variable_neighborhood_search.params.VNSParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.variable_neighborhood_search.params.VNSParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.variable_neighborhood_search.params.VNSParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.variable_neighborhood_search.params.VNSParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.variable_neighborhood_search.params.VNSParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.variable_neighborhood_search.params.VNSParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.variable_neighborhood_search.params.VNSParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.variable_neighborhood_search.params.VNSParams.profit_aware_operators
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.variable_neighborhood_search.params.VNSParams
:canonical: src.policies.variable_neighborhood_search.params.VNSParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.variable_neighborhood_search.params.VNSParams.from_config
```

````

`````
