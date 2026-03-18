# {py:mod}`src.configs.policies.psoda`

```{py:module} src.configs.policies.psoda
```

```{autodoc2-docstring} src.configs.policies.psoda
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DistancePSOConfig <src.configs.policies.psoda.DistancePSOConfig>`
  - ```{autodoc2-docstring} src.configs.policies.psoda.DistancePSOConfig
    :summary:
    ```
````

### API

`````{py:class} DistancePSOConfig
:canonical: src.configs.policies.psoda.DistancePSOConfig

```{autodoc2-docstring} src.configs.policies.psoda.DistancePSOConfig
```

````{py:attribute} population_size
:canonical: src.configs.policies.psoda.DistancePSOConfig.population_size
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.psoda.DistancePSOConfig.population_size
```

````

````{py:attribute} inertia_weight_start
:canonical: src.configs.policies.psoda.DistancePSOConfig.inertia_weight_start
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.configs.policies.psoda.DistancePSOConfig.inertia_weight_start
```

````

````{py:attribute} inertia_weight_end
:canonical: src.configs.policies.psoda.DistancePSOConfig.inertia_weight_end
:type: float
:value: >
   0.4

```{autodoc2-docstring} src.configs.policies.psoda.DistancePSOConfig.inertia_weight_end
```

````

````{py:attribute} cognitive_coef
:canonical: src.configs.policies.psoda.DistancePSOConfig.cognitive_coef
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.psoda.DistancePSOConfig.cognitive_coef
```

````

````{py:attribute} social_coef
:canonical: src.configs.policies.psoda.DistancePSOConfig.social_coef
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.psoda.DistancePSOConfig.social_coef
```

````

````{py:attribute} n_removal
:canonical: src.configs.policies.psoda.DistancePSOConfig.n_removal
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.psoda.DistancePSOConfig.n_removal
```

````

````{py:attribute} velocity_to_mutation_rate
:canonical: src.configs.policies.psoda.DistancePSOConfig.velocity_to_mutation_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.psoda.DistancePSOConfig.velocity_to_mutation_rate
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.psoda.DistancePSOConfig.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.psoda.DistancePSOConfig.max_iterations
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.psoda.DistancePSOConfig.local_search_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.psoda.DistancePSOConfig.local_search_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.psoda.DistancePSOConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.psoda.DistancePSOConfig.time_limit
```

````

````{py:attribute} alpha_profit
:canonical: src.configs.policies.psoda.DistancePSOConfig.alpha_profit
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.psoda.DistancePSOConfig.alpha_profit
```

````

````{py:attribute} beta_will
:canonical: src.configs.policies.psoda.DistancePSOConfig.beta_will
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.psoda.DistancePSOConfig.beta_will
```

````

````{py:attribute} gamma_cost
:canonical: src.configs.policies.psoda.DistancePSOConfig.gamma_cost
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.configs.policies.psoda.DistancePSOConfig.gamma_cost
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.psoda.DistancePSOConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.psoda.DistancePSOConfig.vrpp
```

````

````{py:attribute} seed
:canonical: src.configs.policies.psoda.DistancePSOConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.psoda.DistancePSOConfig.seed
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.psoda.DistancePSOConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.psoda.DistancePSOConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.psoda.DistancePSOConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.psoda.DistancePSOConfig.post_processing
```

````

`````
