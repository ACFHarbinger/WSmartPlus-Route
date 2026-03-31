# {py:mod}`src.configs.policies.bc`

```{py:module} src.configs.policies.bc
```

```{autodoc2-docstring} src.configs.policies.bc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BCConfig <src.configs.policies.bc.BCConfig>`
  - ```{autodoc2-docstring} src.configs.policies.bc.BCConfig
    :summary:
    ```
````

### API

`````{py:class} BCConfig
:canonical: src.configs.policies.bc.BCConfig

```{autodoc2-docstring} src.configs.policies.bc.BCConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.bc.BCConfig.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.configs.policies.bc.BCConfig.time_limit
```

````

````{py:attribute} mip_gap
:canonical: src.configs.policies.bc.BCConfig.mip_gap
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.bc.BCConfig.mip_gap
```

````

````{py:attribute} use_heuristics
:canonical: src.configs.policies.bc.BCConfig.use_heuristics
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.bc.BCConfig.use_heuristics
```

````

````{py:attribute} use_exact_separation
:canonical: src.configs.policies.bc.BCConfig.use_exact_separation
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.bc.BCConfig.use_exact_separation
```

````

````{py:attribute} max_cuts_per_round
:canonical: src.configs.policies.bc.BCConfig.max_cuts_per_round
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.bc.BCConfig.max_cuts_per_round
```

````

````{py:attribute} enable_fractional_capacity_cuts
:canonical: src.configs.policies.bc.BCConfig.enable_fractional_capacity_cuts
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.bc.BCConfig.enable_fractional_capacity_cuts
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.bc.BCConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.bc.BCConfig.profit_aware_operators
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.bc.BCConfig.vrpp
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.bc.BCConfig.vrpp
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.bc.BCConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bc.BCConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.bc.BCConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bc.BCConfig.post_processing
```

````

````{py:attribute} seed
:canonical: src.configs.policies.bc.BCConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bc.BCConfig.seed
```

````

`````
