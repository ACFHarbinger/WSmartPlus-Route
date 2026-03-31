# {py:mod}`src.policies.fast_iterative_localized_optimization.params`

```{py:module} src.policies.fast_iterative_localized_optimization.params
```

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FILOParams <src.policies.fast_iterative_localized_optimization.params.FILOParams>`
  - ```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams
    :summary:
    ```
````

### API

`````{py:class} FILOParams
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams
```

````{py:attribute} time_limit
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.time_limit
```

````

````{py:attribute} max_iterations
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.max_iterations
:type: int
:value: >
   50000

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.max_iterations
```

````

````{py:attribute} initial_temperature_factor
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.initial_temperature_factor
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.initial_temperature_factor
```

````

````{py:attribute} final_temperature_factor
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.final_temperature_factor
:type: float
:value: >
   100.0

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.final_temperature_factor
```

````

````{py:attribute} omega_alpha
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.omega_alpha
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.omega_alpha
```

````

````{py:attribute} shaking_lb_factor
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.shaking_lb_factor
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.shaking_lb_factor
```

````

````{py:attribute} shaking_ub_factor
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.shaking_ub_factor
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.shaking_ub_factor
```

````

````{py:attribute} shaking_lb_intensity
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.shaking_lb_intensity
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.shaking_lb_intensity
```

````

````{py:attribute} shaking_ub_intensity
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.shaking_ub_intensity
:type: float
:value: >
   1.5

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.shaking_ub_intensity
```

````

````{py:attribute} omega_base_multiplier
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.omega_base_multiplier
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.omega_base_multiplier
```

````

````{py:attribute} delta_gamma
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.delta_gamma
:type: float
:value: >
   0.25

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.delta_gamma
```

````

````{py:attribute} gamma_base
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.gamma_base
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.gamma_base
```

````

````{py:attribute} gamma_lambda
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.gamma_lambda
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.gamma_lambda
```

````

````{py:attribute} svc_size
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.svc_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.svc_size
```

````

````{py:attribute} n_cw
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.n_cw
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.n_cw
```

````

````{py:attribute} route_min_sa_temp
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.route_min_sa_temp
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.route_min_sa_temp
```

````

````{py:attribute} local_search_iterations
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.local_search_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.local_search_iterations
```

````

````{py:attribute} seed
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.profit_aware_operators
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.fast_iterative_localized_optimization.params.FILOParams
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.from_config
```

````

`````
