# {py:mod}`src.policies.sine_cosine_algorithm.params`

```{py:module} src.policies.sine_cosine_algorithm.params
```

```{autodoc2-docstring} src.policies.sine_cosine_algorithm.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SCAParams <src.policies.sine_cosine_algorithm.params.SCAParams>`
  - ```{autodoc2-docstring} src.policies.sine_cosine_algorithm.params.SCAParams
    :summary:
    ```
````

### API

`````{py:class} SCAParams
:canonical: src.policies.sine_cosine_algorithm.params.SCAParams

```{autodoc2-docstring} src.policies.sine_cosine_algorithm.params.SCAParams
```

````{py:attribute} pop_size
:canonical: src.policies.sine_cosine_algorithm.params.SCAParams.pop_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.sine_cosine_algorithm.params.SCAParams.pop_size
```

````

````{py:attribute} a_max
:canonical: src.policies.sine_cosine_algorithm.params.SCAParams.a_max
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.policies.sine_cosine_algorithm.params.SCAParams.a_max
```

````

````{py:attribute} max_iterations
:canonical: src.policies.sine_cosine_algorithm.params.SCAParams.max_iterations
:type: int
:value: >
   200

```{autodoc2-docstring} src.policies.sine_cosine_algorithm.params.SCAParams.max_iterations
```

````

````{py:attribute} local_search_iterations
:canonical: src.policies.sine_cosine_algorithm.params.SCAParams.local_search_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.policies.sine_cosine_algorithm.params.SCAParams.local_search_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.sine_cosine_algorithm.params.SCAParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.sine_cosine_algorithm.params.SCAParams.time_limit
```

````

````{py:attribute} vrpp
:canonical: src.policies.sine_cosine_algorithm.params.SCAParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.sine_cosine_algorithm.params.SCAParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.sine_cosine_algorithm.params.SCAParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.sine_cosine_algorithm.params.SCAParams.profit_aware_operators
```

````

````{py:attribute} seed
:canonical: src.policies.sine_cosine_algorithm.params.SCAParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.sine_cosine_algorithm.params.SCAParams.seed
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.sine_cosine_algorithm.params.SCAParams
:canonical: src.policies.sine_cosine_algorithm.params.SCAParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.sine_cosine_algorithm.params.SCAParams.from_config
```

````

`````
