# {py:mod}`src.configs.policies.sa`

```{py:module} src.configs.policies.sa
```

```{autodoc2-docstring} src.configs.policies.sa
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SAConfig <src.configs.policies.sa.SAConfig>`
  - ```{autodoc2-docstring} src.configs.policies.sa.SAConfig
    :summary:
    ```
````

### API

`````{py:class} SAConfig
:canonical: src.configs.policies.sa.SAConfig

```{autodoc2-docstring} src.configs.policies.sa.SAConfig
```

````{py:attribute} initial_temperature
:canonical: src.configs.policies.sa.SAConfig.initial_temperature
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.initial_temperature
```

````

````{py:attribute} cooling_rate
:canonical: src.configs.policies.sa.SAConfig.cooling_rate
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.cooling_rate
```

````

````{py:attribute} min_temperature
:canonical: src.configs.policies.sa.SAConfig.min_temperature
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.min_temperature
```

````

````{py:attribute} iterations_per_temp
:canonical: src.configs.policies.sa.SAConfig.iterations_per_temp
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.iterations_per_temp
```

````

````{py:attribute} nb_granular
:canonical: src.configs.policies.sa.SAConfig.nb_granular
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.nb_granular
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.sa.SAConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.sa.SAConfig.seed
:type: typing.Optional[int]
:value: >
   42

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.sa.SAConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.sa.SAConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.profit_aware_operators
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.sa.SAConfig.mandatory_selection
:type: typing.Optional[typing.List[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.sa.SAConfig.route_improvement
:type: typing.Optional[typing.List[src.configs.policies.other.route_improvement.RouteImprovingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.route_improvement
```

````

````{py:attribute} acceptance
:canonical: src.configs.policies.sa.SAConfig.acceptance
:type: src.configs.policies.other.acceptance_criteria.AcceptanceConfig
:value: >
   'AcceptanceConfig(...)'

```{autodoc2-docstring} src.configs.policies.sa.SAConfig.acceptance
```

````

`````
