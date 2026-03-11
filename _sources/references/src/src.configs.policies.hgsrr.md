# {py:mod}`src.configs.policies.hgsrr`

```{py:module} src.configs.policies.hgsrr
```

```{autodoc2-docstring} src.configs.policies.hgsrr
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSRRConfig <src.configs.policies.hgsrr.HGSRRConfig>`
  - ```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig
    :summary:
    ```
````

### API

`````{py:class} HGSRRConfig
:canonical: src.configs.policies.hgsrr.HGSRRConfig

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig
```

````{py:attribute} time_limit
:canonical: src.configs.policies.hgsrr.HGSRRConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.hgsrr.HGSRRConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.seed
```

````

````{py:attribute} population_size
:canonical: src.configs.policies.hgsrr.HGSRRConfig.population_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.population_size
```

````

````{py:attribute} elite_size
:canonical: src.configs.policies.hgsrr.HGSRRConfig.elite_size
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.elite_size
```

````

````{py:attribute} mutation_rate
:canonical: src.configs.policies.hgsrr.HGSRRConfig.mutation_rate
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.mutation_rate
```

````

````{py:attribute} crossover_rate
:canonical: src.configs.policies.hgsrr.HGSRRConfig.crossover_rate
:type: float
:value: >
   0.7

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.crossover_rate
```

````

````{py:attribute} n_generations
:canonical: src.configs.policies.hgsrr.HGSRRConfig.n_generations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.n_generations
```

````

````{py:attribute} alpha_diversity
:canonical: src.configs.policies.hgsrr.HGSRRConfig.alpha_diversity
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.alpha_diversity
```

````

````{py:attribute} min_diversity
:canonical: src.configs.policies.hgsrr.HGSRRConfig.min_diversity
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.min_diversity
```

````

````{py:attribute} diversity_change_rate
:canonical: src.configs.policies.hgsrr.HGSRRConfig.diversity_change_rate
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.diversity_change_rate
```

````

````{py:attribute} no_improvement_threshold
:canonical: src.configs.policies.hgsrr.HGSRRConfig.no_improvement_threshold
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.no_improvement_threshold
```

````

````{py:attribute} survivor_threshold
:canonical: src.configs.policies.hgsrr.HGSRRConfig.survivor_threshold
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.survivor_threshold
```

````

````{py:attribute} neighbor_list_size
:canonical: src.configs.policies.hgsrr.HGSRRConfig.neighbor_list_size
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.neighbor_list_size
```

````

````{py:attribute} max_vehicles
:canonical: src.configs.policies.hgsrr.HGSRRConfig.max_vehicles
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.max_vehicles
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.hgsrr.HGSRRConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.vrpp
```

````

````{py:attribute} min_removal_pct
:canonical: src.configs.policies.hgsrr.HGSRRConfig.min_removal_pct
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.min_removal_pct
```

````

````{py:attribute} max_removal_pct
:canonical: src.configs.policies.hgsrr.HGSRRConfig.max_removal_pct
:type: float
:value: >
   0.4

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.max_removal_pct
```

````

````{py:attribute} noise_factor
:canonical: src.configs.policies.hgsrr.HGSRRConfig.noise_factor
:type: float
:value: >
   0.015

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.noise_factor
```

````

````{py:attribute} reaction_factor
:canonical: src.configs.policies.hgsrr.HGSRRConfig.reaction_factor
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.reaction_factor
```

````

````{py:attribute} decay_parameter
:canonical: src.configs.policies.hgsrr.HGSRRConfig.decay_parameter
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.decay_parameter
```

````

````{py:attribute} destroy_operators
:canonical: src.configs.policies.hgsrr.HGSRRConfig.destroy_operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.destroy_operators
```

````

````{py:attribute} repair_operators
:canonical: src.configs.policies.hgsrr.HGSRRConfig.repair_operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.repair_operators
```

````

````{py:attribute} operator_decay_rate
:canonical: src.configs.policies.hgsrr.HGSRRConfig.operator_decay_rate
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.operator_decay_rate
```

````

````{py:attribute} score_sigma_1
:canonical: src.configs.policies.hgsrr.HGSRRConfig.score_sigma_1
:type: float
:value: >
   33.0

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.score_sigma_1
```

````

````{py:attribute} score_sigma_2
:canonical: src.configs.policies.hgsrr.HGSRRConfig.score_sigma_2
:type: float
:value: >
   9.0

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.score_sigma_2
```

````

````{py:attribute} score_sigma_3
:canonical: src.configs.policies.hgsrr.HGSRRConfig.score_sigma_3
:type: float
:value: >
   3.0

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.score_sigma_3
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.hgsrr.HGSRRConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.hgsrr.HGSRRConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hgsrr.HGSRRConfig.post_processing
```

````

`````
