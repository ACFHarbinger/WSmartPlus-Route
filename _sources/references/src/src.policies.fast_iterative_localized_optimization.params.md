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

````{py:attribute} delta_gamma
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.delta_gamma
:type: float
:value: >
   0.1

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

````{py:attribute} omega_base_multiplier
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.omega_base_multiplier
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.omega_base_multiplier
```

````

````{py:attribute} seed
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.seed
:type: int
:value: >
   42

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.seed
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

````{py:method} from_config(config: logic.src.configs.policies.filo.FILOConfig) -> src.policies.fast_iterative_localized_optimization.params.FILOParams
:canonical: src.policies.fast_iterative_localized_optimization.params.FILOParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.params.FILOParams.from_config
```

````

`````
