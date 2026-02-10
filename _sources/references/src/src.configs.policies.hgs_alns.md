# {py:mod}`src.configs.policies.hgs_alns`

```{py:module} src.configs.policies.hgs_alns
```

```{autodoc2-docstring} src.configs.policies.hgs_alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSALNSConfig <src.configs.policies.hgs_alns.HGSALNSConfig>`
  - ```{autodoc2-docstring} src.configs.policies.hgs_alns.HGSALNSConfig
    :summary:
    ```
````

### API

`````{py:class} HGSALNSConfig
:canonical: src.configs.policies.hgs_alns.HGSALNSConfig

```{autodoc2-docstring} src.configs.policies.hgs_alns.HGSALNSConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.hgs_alns.HGSALNSConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.hgs_alns.HGSALNSConfig.time_limit
```

````

````{py:attribute} population_size
:canonical: src.configs.policies.hgs_alns.HGSALNSConfig.population_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.hgs_alns.HGSALNSConfig.population_size
```

````

````{py:attribute} elite_size
:canonical: src.configs.policies.hgs_alns.HGSALNSConfig.elite_size
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.hgs_alns.HGSALNSConfig.elite_size
```

````

````{py:attribute} mutation_rate
:canonical: src.configs.policies.hgs_alns.HGSALNSConfig.mutation_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.hgs_alns.HGSALNSConfig.mutation_rate
```

````

````{py:attribute} alns_education_iterations
:canonical: src.configs.policies.hgs_alns.HGSALNSConfig.alns_education_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.hgs_alns.HGSALNSConfig.alns_education_iterations
```

````

````{py:attribute} n_generations
:canonical: src.configs.policies.hgs_alns.HGSALNSConfig.n_generations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.hgs_alns.HGSALNSConfig.n_generations
```

````

````{py:attribute} max_vehicles
:canonical: src.configs.policies.hgs_alns.HGSALNSConfig.max_vehicles
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.policies.hgs_alns.HGSALNSConfig.max_vehicles
```

````

````{py:attribute} engine
:canonical: src.configs.policies.hgs_alns.HGSALNSConfig.engine
:type: str
:value: >
   'hgs_alns'

```{autodoc2-docstring} src.configs.policies.hgs_alns.HGSALNSConfig.engine
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.hgs_alns.HGSALNSConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hgs_alns.HGSALNSConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.hgs_alns.HGSALNSConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hgs_alns.HGSALNSConfig.post_processing
```

````

`````
