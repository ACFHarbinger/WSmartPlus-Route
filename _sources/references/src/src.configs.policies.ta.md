# {py:mod}`src.configs.policies.ta`

```{py:module} src.configs.policies.ta
```

```{autodoc2-docstring} src.configs.policies.ta
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TAConfig <src.configs.policies.ta.TAConfig>`
  - ```{autodoc2-docstring} src.configs.policies.ta.TAConfig
    :summary:
    ```
````

### API

`````{py:class} TAConfig
:canonical: src.configs.policies.ta.TAConfig

```{autodoc2-docstring} src.configs.policies.ta.TAConfig
```

````{py:attribute} engine
:canonical: src.configs.policies.ta.TAConfig.engine
:type: str
:value: >
   'ta'

```{autodoc2-docstring} src.configs.policies.ta.TAConfig.engine
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.ta.TAConfig.max_iterations
:type: int
:value: >
   1000

```{autodoc2-docstring} src.configs.policies.ta.TAConfig.max_iterations
```

````

````{py:attribute} initial_threshold
:canonical: src.configs.policies.ta.TAConfig.initial_threshold
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.policies.ta.TAConfig.initial_threshold
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.ta.TAConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.ta.TAConfig.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.configs.policies.ta.TAConfig.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.ta.TAConfig.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.ta.TAConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.ta.TAConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.ta.TAConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.ta.TAConfig.seed
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.ta.TAConfig.profit_aware_operators
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ta.TAConfig.profit_aware_operators
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.ta.TAConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.ta.TAConfig.vrpp
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.ta.TAConfig.must_go
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ta.TAConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.ta.TAConfig.post_processing
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.ta.TAConfig.post_processing
```

````

`````
