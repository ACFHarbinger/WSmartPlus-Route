# {py:mod}`src.configs.policies.oba`

```{py:module} src.configs.policies.oba
```

```{autodoc2-docstring} src.configs.policies.oba
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OBAConfig <src.configs.policies.oba.OBAConfig>`
  - ```{autodoc2-docstring} src.configs.policies.oba.OBAConfig
    :summary:
    ```
````

### API

`````{py:class} OBAConfig
:canonical: src.configs.policies.oba.OBAConfig

```{autodoc2-docstring} src.configs.policies.oba.OBAConfig
```

````{py:attribute} dilation
:canonical: src.configs.policies.oba.OBAConfig.dilation
:type: float
:value: >
   5.0

```{autodoc2-docstring} src.configs.policies.oba.OBAConfig.dilation
```

````

````{py:attribute} contraction
:canonical: src.configs.policies.oba.OBAConfig.contraction
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.oba.OBAConfig.contraction
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.oba.OBAConfig.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.oba.OBAConfig.max_iterations
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.oba.OBAConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.oba.OBAConfig.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.configs.policies.oba.OBAConfig.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.oba.OBAConfig.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.oba.OBAConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.oba.OBAConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.oba.OBAConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.oba.OBAConfig.seed
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.oba.OBAConfig.profit_aware_operators
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.oba.OBAConfig.profit_aware_operators
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.oba.OBAConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.oba.OBAConfig.vrpp
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.oba.OBAConfig.mandatory_selection
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.oba.OBAConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.oba.OBAConfig.route_improvement
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.oba.OBAConfig.route_improvement
```

````

`````
