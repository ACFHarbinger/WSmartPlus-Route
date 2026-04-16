# {py:mod}`src.configs.policies.rts`

```{py:module} src.configs.policies.rts
```

```{autodoc2-docstring} src.configs.policies.rts
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RTSConfig <src.configs.policies.rts.RTSConfig>`
  - ```{autodoc2-docstring} src.configs.policies.rts.RTSConfig
    :summary:
    ```
````

### API

`````{py:class} RTSConfig
:canonical: src.configs.policies.rts.RTSConfig

```{autodoc2-docstring} src.configs.policies.rts.RTSConfig
```

````{py:attribute} initial_tenure
:canonical: src.configs.policies.rts.RTSConfig.initial_tenure
:type: int
:value: >
   7

```{autodoc2-docstring} src.configs.policies.rts.RTSConfig.initial_tenure
```

````

````{py:attribute} min_tenure
:canonical: src.configs.policies.rts.RTSConfig.min_tenure
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.rts.RTSConfig.min_tenure
```

````

````{py:attribute} max_tenure
:canonical: src.configs.policies.rts.RTSConfig.max_tenure
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.rts.RTSConfig.max_tenure
```

````

````{py:attribute} tenure_increase
:canonical: src.configs.policies.rts.RTSConfig.tenure_increase
:type: float
:value: >
   1.5

```{autodoc2-docstring} src.configs.policies.rts.RTSConfig.tenure_increase
```

````

````{py:attribute} tenure_decrease
:canonical: src.configs.policies.rts.RTSConfig.tenure_decrease
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.configs.policies.rts.RTSConfig.tenure_decrease
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.rts.RTSConfig.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.rts.RTSConfig.max_iterations
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.rts.RTSConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.rts.RTSConfig.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.configs.policies.rts.RTSConfig.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.rts.RTSConfig.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.rts.RTSConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.rts.RTSConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.rts.RTSConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.rts.RTSConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.rts.RTSConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.rts.RTSConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.rts.RTSConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.rts.RTSConfig.profit_aware_operators
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.rts.RTSConfig.mandatory_selection
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.rts.RTSConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.rts.RTSConfig.route_improvement
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.rts.RTSConfig.route_improvement
```

````

````{py:attribute} acceptance
:canonical: src.configs.policies.rts.RTSConfig.acceptance
:type: src.configs.policies.other.acceptance_criteria.AcceptanceConfig
:value: >
   'AcceptanceConfig(...)'

```{autodoc2-docstring} src.configs.policies.rts.RTSConfig.acceptance
```

````

`````
