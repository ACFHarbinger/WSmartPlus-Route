# {py:mod}`src.configs.policies.filo`

```{py:module} src.configs.policies.filo
```

```{autodoc2-docstring} src.configs.policies.filo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FILOConfig <src.configs.policies.filo.FILOConfig>`
  - ```{autodoc2-docstring} src.configs.policies.filo.FILOConfig
    :summary:
    ```
````

### API

`````{py:class} FILOConfig
:canonical: src.configs.policies.filo.FILOConfig

Bases: {py:obj}`src.configs.policies.abc.ABCConfig`

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.filo.FILOConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig.time_limit
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.filo.FILOConfig.max_iterations
:type: int
:value: >
   50000

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig.max_iterations
```

````

````{py:attribute} initial_temperature_factor
:canonical: src.configs.policies.filo.FILOConfig.initial_temperature_factor
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig.initial_temperature_factor
```

````

````{py:attribute} final_temperature_factor
:canonical: src.configs.policies.filo.FILOConfig.final_temperature_factor
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig.final_temperature_factor
```

````

````{py:attribute} shaking_lb_factor
:canonical: src.configs.policies.filo.FILOConfig.shaking_lb_factor
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig.shaking_lb_factor
```

````

````{py:attribute} shaking_ub_factor
:canonical: src.configs.policies.filo.FILOConfig.shaking_ub_factor
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig.shaking_ub_factor
```

````

````{py:attribute} shaking_lb_intensity
:canonical: src.configs.policies.filo.FILOConfig.shaking_lb_intensity
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig.shaking_lb_intensity
```

````

````{py:attribute} shaking_ub_intensity
:canonical: src.configs.policies.filo.FILOConfig.shaking_ub_intensity
:type: float
:value: >
   1.5

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig.shaking_ub_intensity
```

````

````{py:attribute} omega_alpha
:canonical: src.configs.policies.filo.FILOConfig.omega_alpha
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig.omega_alpha
```

````

````{py:attribute} omega_base_multiplier
:canonical: src.configs.policies.filo.FILOConfig.omega_base_multiplier
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig.omega_base_multiplier
```

````

````{py:attribute} delta_gamma
:canonical: src.configs.policies.filo.FILOConfig.delta_gamma
:type: float
:value: >
   0.25

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig.delta_gamma
```

````

````{py:attribute} gamma_base
:canonical: src.configs.policies.filo.FILOConfig.gamma_base
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig.gamma_base
```

````

````{py:attribute} gamma_lambda
:canonical: src.configs.policies.filo.FILOConfig.gamma_lambda
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig.gamma_lambda
```

````

````{py:attribute} svc_size
:canonical: src.configs.policies.filo.FILOConfig.svc_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig.svc_size
```

````

````{py:attribute} n_cw
:canonical: src.configs.policies.filo.FILOConfig.n_cw
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig.n_cw
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.filo.FILOConfig.local_search_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig.local_search_iterations
```

````

````{py:attribute} seed
:canonical: src.configs.policies.filo.FILOConfig.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig.seed
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.filo.FILOConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.filo.FILOConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.filo.FILOConfig.profit_aware_operators
```

````

`````
