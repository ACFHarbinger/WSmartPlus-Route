# {py:mod}`src.policies.threshold_accepting.params`

```{py:module} src.policies.threshold_accepting.params
```

```{autodoc2-docstring} src.policies.threshold_accepting.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TAParams <src.policies.threshold_accepting.params.TAParams>`
  - ```{autodoc2-docstring} src.policies.threshold_accepting.params.TAParams
    :summary:
    ```
````

### API

`````{py:class} TAParams
:canonical: src.policies.threshold_accepting.params.TAParams

```{autodoc2-docstring} src.policies.threshold_accepting.params.TAParams
```

````{py:attribute} max_iterations
:canonical: src.policies.threshold_accepting.params.TAParams.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.policies.threshold_accepting.params.TAParams.max_iterations
```

````

````{py:attribute} initial_threshold
:canonical: src.policies.threshold_accepting.params.TAParams.initial_threshold
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.policies.threshold_accepting.params.TAParams.initial_threshold
```

````

````{py:attribute} time_limit
:canonical: src.policies.threshold_accepting.params.TAParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.threshold_accepting.params.TAParams.time_limit
```

````

````{py:attribute} n_removal
:canonical: src.policies.threshold_accepting.params.TAParams.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.policies.threshold_accepting.params.TAParams.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.policies.threshold_accepting.params.TAParams.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.threshold_accepting.params.TAParams.n_llh
```

````

````{py:attribute} seed
:canonical: src.policies.threshold_accepting.params.TAParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.threshold_accepting.params.TAParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.threshold_accepting.params.TAParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.threshold_accepting.params.TAParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.threshold_accepting.params.TAParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.threshold_accepting.params.TAParams.profit_aware_operators
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.threshold_accepting.params.TAParams
:canonical: src.policies.threshold_accepting.params.TAParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.threshold_accepting.params.TAParams.from_config
```

````

`````
