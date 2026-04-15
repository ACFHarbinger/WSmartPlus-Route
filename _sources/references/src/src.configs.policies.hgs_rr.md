# {py:mod}`src.configs.policies.hgs_rr`

```{py:module} src.configs.policies.hgs_rr
```

```{autodoc2-docstring} src.configs.policies.hgs_rr
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSRRConfig <src.configs.policies.hgs_rr.HGSRRConfig>`
  - ```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig
    :summary:
    ```
````

### API

`````{py:class} HGSRRConfig
:canonical: src.configs.policies.hgs_rr.HGSRRConfig

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig
```

````{py:attribute} restart_timer
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.restart_timer
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.restart_timer
```

````

````{py:attribute} time_limit
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.time_limit
:type: float
:value: >
   60.0

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.seed
```

````

````{py:attribute} population_size
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.population_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.population_size
```

````

````{py:attribute} elite_size
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.elite_size
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.elite_size
```

````

````{py:attribute} mutation_rate
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.mutation_rate
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.mutation_rate
```

````

````{py:attribute} crossover_rate
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.crossover_rate
:type: float
:value: >
   0.7

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.crossover_rate
```

````

````{py:attribute} n_iterations_no_improvement
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.n_iterations_no_improvement
:type: int
:value: >
   20000

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.n_iterations_no_improvement
```

````

````{py:attribute} no_improvement_threshold
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.no_improvement_threshold
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.no_improvement_threshold
```

````

````{py:attribute} survivor_threshold
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.survivor_threshold
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.survivor_threshold
```

````

````{py:attribute} neighbor_list_size
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.neighbor_list_size
:type: int
:value: >
   10

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.neighbor_list_size
```

````

````{py:attribute} max_vehicles
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.max_vehicles
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.max_vehicles
```

````

````{py:attribute} vrpp
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.profit_aware_operators
```

````

````{py:attribute} min_removal_pct
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.min_removal_pct
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.min_removal_pct
```

````

````{py:attribute} max_removal_pct
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.max_removal_pct
:type: float
:value: >
   0.4

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.max_removal_pct
```

````

````{py:attribute} noise_factor
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.noise_factor
:type: float
:value: >
   0.015

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.noise_factor
```

````

````{py:attribute} reaction_factor
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.reaction_factor
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.reaction_factor
```

````

````{py:attribute} decay_parameter
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.decay_parameter
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.decay_parameter
```

````

````{py:attribute} destroy_operators
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.destroy_operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.destroy_operators
```

````

````{py:attribute} repair_operators
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.repair_operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.repair_operators
```

````

````{py:attribute} operator_decay_rate
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.operator_decay_rate
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.operator_decay_rate
```

````

````{py:attribute} score_sigma_1
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.score_sigma_1
:type: float
:value: >
   33.0

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.score_sigma_1
```

````

````{py:attribute} score_sigma_2
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.score_sigma_2
:type: float
:value: >
   9.0

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.score_sigma_2
```

````

````{py:attribute} score_sigma_3
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.score_sigma_3
:type: float
:value: >
   3.0

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.score_sigma_3
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.hgs_rr.HGSRRConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hgs_rr.HGSRRConfig.post_processing
```

````

`````
