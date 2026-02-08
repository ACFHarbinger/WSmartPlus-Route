# {py:mod}`src.configs.policies.hgs`

```{py:module} src.configs.policies.hgs
```

```{autodoc2-docstring} src.configs.policies.hgs
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSConfig <src.configs.policies.hgs.HGSConfig>`
  - ```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig
    :summary:
    ```
````

### API

`````{py:class} HGSConfig
:canonical: src.configs.policies.hgs.HGSConfig

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.hgs.HGSConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.time_limit
```

````

````{py:attribute} population_size
:canonical: src.configs.policies.hgs.HGSConfig.population_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.population_size
```

````

````{py:attribute} elite_size
:canonical: src.configs.policies.hgs.HGSConfig.elite_size
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.elite_size
```

````

````{py:attribute} mutation_rate
:canonical: src.configs.policies.hgs.HGSConfig.mutation_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.mutation_rate
```

````

````{py:attribute} n_generations
:canonical: src.configs.policies.hgs.HGSConfig.n_generations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.n_generations
```

````

````{py:attribute} max_vehicles
:canonical: src.configs.policies.hgs.HGSConfig.max_vehicles
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.max_vehicles
```

````

````{py:attribute} engine
:canonical: src.configs.policies.hgs.HGSConfig.engine
:type: str
:value: >
   'custom'

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.engine
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.hgs.HGSConfig.must_go
:type: typing.Optional[typing.List[src.configs.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.hgs.HGSConfig.post_processing
:type: typing.Optional[typing.List[src.configs.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.post_processing
```

````

`````
