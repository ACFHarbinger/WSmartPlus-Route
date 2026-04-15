# {py:mod}`src.configs.policies.hms`

```{py:module} src.configs.policies.hms
```

```{autodoc2-docstring} src.configs.policies.hms
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HybridMemeticSearchConfig <src.configs.policies.hms.HybridMemeticSearchConfig>`
  - ```{autodoc2-docstring} src.configs.policies.hms.HybridMemeticSearchConfig
    :summary:
    ```
````

### API

`````{py:class} HybridMemeticSearchConfig
:canonical: src.configs.policies.hms.HybridMemeticSearchConfig

```{autodoc2-docstring} src.configs.policies.hms.HybridMemeticSearchConfig
```

````{py:attribute} n_removal
:canonical: src.configs.policies.hms.HybridMemeticSearchConfig.n_removal
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.hms.HybridMemeticSearchConfig.n_removal
```

````

````{py:attribute} population_size
:canonical: src.configs.policies.hms.HybridMemeticSearchConfig.population_size
:type: int
:value: >
   30

```{autodoc2-docstring} src.configs.policies.hms.HybridMemeticSearchConfig.population_size
```

````

````{py:attribute} max_generations
:canonical: src.configs.policies.hms.HybridMemeticSearchConfig.max_generations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.hms.HybridMemeticSearchConfig.max_generations
```

````

````{py:attribute} substitution_rate
:canonical: src.configs.policies.hms.HybridMemeticSearchConfig.substitution_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.hms.HybridMemeticSearchConfig.substitution_rate
```

````

````{py:attribute} crossover_rate
:canonical: src.configs.policies.hms.HybridMemeticSearchConfig.crossover_rate
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.configs.policies.hms.HybridMemeticSearchConfig.crossover_rate
```

````

````{py:attribute} mutation_rate
:canonical: src.configs.policies.hms.HybridMemeticSearchConfig.mutation_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.hms.HybridMemeticSearchConfig.mutation_rate
```

````

````{py:attribute} elitism_count
:canonical: src.configs.policies.hms.HybridMemeticSearchConfig.elitism_count
:type: int
:value: >
   3

```{autodoc2-docstring} src.configs.policies.hms.HybridMemeticSearchConfig.elitism_count
```

````

````{py:attribute} aco_init_iterations
:canonical: src.configs.policies.hms.HybridMemeticSearchConfig.aco_init_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.hms.HybridMemeticSearchConfig.aco_init_iterations
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.hms.HybridMemeticSearchConfig.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.configs.policies.hms.HybridMemeticSearchConfig.time_limit
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.hms.HybridMemeticSearchConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.hms.HybridMemeticSearchConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.hms.HybridMemeticSearchConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.hms.HybridMemeticSearchConfig.profit_aware_operators
```

````

````{py:attribute} seed
:canonical: src.configs.policies.hms.HybridMemeticSearchConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hms.HybridMemeticSearchConfig.seed
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.hms.HybridMemeticSearchConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hms.HybridMemeticSearchConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.hms.HybridMemeticSearchConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hms.HybridMemeticSearchConfig.post_processing
```

````

`````
