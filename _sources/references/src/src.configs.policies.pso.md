# {py:mod}`src.configs.policies.pso`

```{py:module} src.configs.policies.pso
```

```{autodoc2-docstring} src.configs.policies.pso
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PSOConfig <src.configs.policies.pso.PSOConfig>`
  - ```{autodoc2-docstring} src.configs.policies.pso.PSOConfig
    :summary:
    ```
````

### API

`````{py:class} PSOConfig
:canonical: src.configs.policies.pso.PSOConfig

```{autodoc2-docstring} src.configs.policies.pso.PSOConfig
```

````{py:attribute} pop_size
:canonical: src.configs.policies.pso.PSOConfig.pop_size
:type: int
:value: >
   30

```{autodoc2-docstring} src.configs.policies.pso.PSOConfig.pop_size
```

````

````{py:attribute} inertia_weight_start
:canonical: src.configs.policies.pso.PSOConfig.inertia_weight_start
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.configs.policies.pso.PSOConfig.inertia_weight_start
```

````

````{py:attribute} inertia_weight_end
:canonical: src.configs.policies.pso.PSOConfig.inertia_weight_end
:type: float
:value: >
   0.4

```{autodoc2-docstring} src.configs.policies.pso.PSOConfig.inertia_weight_end
```

````

````{py:attribute} cognitive_coef
:canonical: src.configs.policies.pso.PSOConfig.cognitive_coef
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.pso.PSOConfig.cognitive_coef
```

````

````{py:attribute} social_coef
:canonical: src.configs.policies.pso.PSOConfig.social_coef
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.pso.PSOConfig.social_coef
```

````

````{py:attribute} position_min
:canonical: src.configs.policies.pso.PSOConfig.position_min
:type: float
:value: >
   None

```{autodoc2-docstring} src.configs.policies.pso.PSOConfig.position_min
```

````

````{py:attribute} position_max
:canonical: src.configs.policies.pso.PSOConfig.position_max
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.pso.PSOConfig.position_max
```

````

````{py:attribute} velocity_max
:canonical: src.configs.policies.pso.PSOConfig.velocity_max
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.pso.PSOConfig.velocity_max
```

````

````{py:attribute} max_iterations
:canonical: src.configs.policies.pso.PSOConfig.max_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.pso.PSOConfig.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.pso.PSOConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.pso.PSOConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.pso.PSOConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.pso.PSOConfig.seed
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.pso.PSOConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.pso.PSOConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.pso.PSOConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.pso.PSOConfig.post_processing
```

````

`````
