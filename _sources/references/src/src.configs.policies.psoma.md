# {py:mod}`src.configs.policies.psoma`

```{py:module} src.configs.policies.psoma
```

```{autodoc2-docstring} src.configs.policies.psoma
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PSOMAConfig <src.configs.policies.psoma.PSOMAConfig>`
  - ```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig
    :summary:
    ```
````

### API

`````{py:class} PSOMAConfig
:canonical: src.configs.policies.psoma.PSOMAConfig

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig
```

````{py:attribute} pop_size
:canonical: src.configs.policies.psoma.PSOMAConfig.pop_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.pop_size
```

````

````{py:attribute} omega
:canonical: src.configs.policies.psoma.PSOMAConfig.omega
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.omega
```

````

````{py:attribute} c1
:canonical: src.configs.policies.psoma.PSOMAConfig.c1
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.c1
```

````

````{py:attribute} c2
:canonical: src.configs.policies.psoma.PSOMAConfig.c2
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.c2
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.psoma.PSOMAConfig.max_iterations
:type: int
:value: >
   200

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.max_iterations
```

````

````{py:attribute} x_min
:canonical: src.configs.policies.psoma.PSOMAConfig.x_min
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.x_min
```

````

````{py:attribute} x_max
:canonical: src.configs.policies.psoma.PSOMAConfig.x_max
:type: float
:value: >
   4.0

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.x_max
```

````

````{py:attribute} v_min
:canonical: src.configs.policies.psoma.PSOMAConfig.v_min
:type: float
:value: >
   None

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.v_min
```

````

````{py:attribute} v_max
:canonical: src.configs.policies.psoma.PSOMAConfig.v_max
:type: float
:value: >
   4.0

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.v_max
```

````

````{py:attribute} L
:canonical: src.configs.policies.psoma.PSOMAConfig.L
:type: int
:value: >
   30

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.L
```

````

````{py:attribute} T0
:canonical: src.configs.policies.psoma.PSOMAConfig.T0
:type: float
:value: >
   3.0

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.T0
```

````

````{py:attribute} lambda_cooling
:canonical: src.configs.policies.psoma.PSOMAConfig.lambda_cooling
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.lambda_cooling
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.psoma.PSOMAConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.time_limit
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.psoma.PSOMAConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.psoma.PSOMAConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.profit_aware_operators
```

````

````{py:attribute} seed
:canonical: src.configs.policies.psoma.PSOMAConfig.seed
:type: typing.Optional[int]
:value: >
   42

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.seed
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.psoma.PSOMAConfig.mandatory_selection
:type: typing.Optional[typing.List[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.psoma.PSOMAConfig.route_improvement
:type: typing.Optional[typing.List[src.configs.policies.other.route_improvement.RouteImprovingConfig]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.route_improvement
```

````

````{py:attribute} acceptance_criterion
:canonical: src.configs.policies.psoma.PSOMAConfig.acceptance_criterion
:type: src.configs.policies.other.acceptance_criteria.AcceptanceConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.psoma.PSOMAConfig.acceptance_criterion
```

````

`````
