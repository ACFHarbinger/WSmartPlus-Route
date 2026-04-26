# {py:mod}`src.configs.policies.alns`

```{py:module} src.configs.policies.alns
```

```{autodoc2-docstring} src.configs.policies.alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ALNSConfig <src.configs.policies.alns.ALNSConfig>`
  - ```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig
    :summary:
    ```
````

### API

`````{py:class} ALNSConfig
:canonical: src.configs.policies.alns.ALNSConfig

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.alns.ALNSConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.alns.ALNSConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.seed
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.alns.ALNSConfig.max_iterations
:type: int
:value: >
   5000

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.max_iterations
```

````

````{py:attribute} start_temp
:canonical: src.configs.policies.alns.ALNSConfig.start_temp
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.start_temp
```

````

````{py:attribute} cooling_rate
:canonical: src.configs.policies.alns.ALNSConfig.cooling_rate
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.cooling_rate
```

````

````{py:attribute} reaction_factor
:canonical: src.configs.policies.alns.ALNSConfig.reaction_factor
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.reaction_factor
```

````

````{py:attribute} min_removal
:canonical: src.configs.policies.alns.ALNSConfig.min_removal
:type: int
:value: >
   1

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.min_removal
```

````

````{py:attribute} start_temp_control
:canonical: src.configs.policies.alns.ALNSConfig.start_temp_control
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.start_temp_control
```

````

````{py:attribute} xi
:canonical: src.configs.policies.alns.ALNSConfig.xi
:type: float
:value: >
   0.4

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.xi
```

````

````{py:attribute} engine
:canonical: src.configs.policies.alns.ALNSConfig.engine
:type: str
:value: >
   'custom'

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.engine
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.alns.ALNSConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.alns.ALNSConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.profit_aware_operators
```

````

````{py:attribute} extended_operators
:canonical: src.configs.policies.alns.ALNSConfig.extended_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.extended_operators
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.alns.ALNSConfig.mandatory_selection
:type: typing.Optional[typing.List[src.configs.policies.other.mandatory_selection.MandatorySelectionConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.alns.ALNSConfig.route_improvement
:type: typing.Optional[typing.List[src.configs.policies.other.route_improvement.RouteImprovingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.route_improvement
```

````

````{py:attribute} acceptance_criterion
:canonical: src.configs.policies.alns.ALNSConfig.acceptance_criterion
:type: src.configs.policies.other.acceptance_criteria.AcceptanceConfig
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.alns.ALNSConfig.acceptance_criterion
```

````

`````
