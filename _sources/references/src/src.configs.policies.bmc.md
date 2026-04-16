# {py:mod}`src.configs.policies.bmc`

```{py:module} src.configs.policies.bmc
```

```{autodoc2-docstring} src.configs.policies.bmc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BMCConfig <src.configs.policies.bmc.BMCConfig>`
  - ```{autodoc2-docstring} src.configs.policies.bmc.BMCConfig
    :summary:
    ```
````

### API

`````{py:class} BMCConfig
:canonical: src.configs.policies.bmc.BMCConfig

```{autodoc2-docstring} src.configs.policies.bmc.BMCConfig
```

````{py:attribute} initial_temp
:canonical: src.configs.policies.bmc.BMCConfig.initial_temp
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.policies.bmc.BMCConfig.initial_temp
```

````

````{py:attribute} alpha
:canonical: src.configs.policies.bmc.BMCConfig.alpha
:type: float
:value: >
   0.995

```{autodoc2-docstring} src.configs.policies.bmc.BMCConfig.alpha
```

````

````{py:attribute} min_temp
:canonical: src.configs.policies.bmc.BMCConfig.min_temp
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.configs.policies.bmc.BMCConfig.min_temp
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.bmc.BMCConfig.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.bmc.BMCConfig.max_iterations
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.bmc.BMCConfig.n_removal
:type: int
:value: >
   2

```{autodoc2-docstring} src.configs.policies.bmc.BMCConfig.n_removal
```

````

````{py:attribute} n_llh
:canonical: src.configs.policies.bmc.BMCConfig.n_llh
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.bmc.BMCConfig.n_llh
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.bmc.BMCConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.bmc.BMCConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.bmc.BMCConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.bmc.BMCConfig.seed
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.bmc.BMCConfig.profit_aware_operators
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.bmc.BMCConfig.profit_aware_operators
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.bmc.BMCConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.bmc.BMCConfig.vrpp
```

````

````{py:attribute} mandatory_selection
:canonical: src.configs.policies.bmc.BMCConfig.mandatory_selection
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.bmc.BMCConfig.mandatory_selection
```

````

````{py:attribute} route_improvement
:canonical: src.configs.policies.bmc.BMCConfig.route_improvement
:type: typing.Optional[typing.List[typing.Any]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.bmc.BMCConfig.route_improvement
```

````

`````
