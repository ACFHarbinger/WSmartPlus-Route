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
   0.0

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.time_limit
```

````

````{py:attribute} seed
:canonical: src.configs.policies.hgs.HGSConfig.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.seed
```

````

````{py:attribute} mu
:canonical: src.configs.policies.hgs.HGSConfig.mu
:type: int
:value: >
   25

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.mu
```

````

````{py:attribute} n_offspring
:canonical: src.configs.policies.hgs.HGSConfig.n_offspring
:type: int
:value: >
   40

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.n_offspring
```

````

````{py:attribute} nb_elite
:canonical: src.configs.policies.hgs.HGSConfig.nb_elite
:type: int
:value: >
   4

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.nb_elite
```

````

````{py:attribute} nb_close
:canonical: src.configs.policies.hgs.HGSConfig.nb_close
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.nb_close
```

````

````{py:attribute} nb_granular
:canonical: src.configs.policies.hgs.HGSConfig.nb_granular
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.nb_granular
```

````

````{py:attribute} target_feasible
:canonical: src.configs.policies.hgs.HGSConfig.target_feasible
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.target_feasible
```

````

````{py:attribute} n_iterations_no_improvement
:canonical: src.configs.policies.hgs.HGSConfig.n_iterations_no_improvement
:type: int
:value: >
   20000

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.n_iterations_no_improvement
```

````

````{py:attribute} mutation_rate
:canonical: src.configs.policies.hgs.HGSConfig.mutation_rate
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.mutation_rate
```

````

````{py:attribute} repair_probability
:canonical: src.configs.policies.hgs.HGSConfig.repair_probability
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.repair_probability
```

````

````{py:attribute} crossover_rate
:canonical: src.configs.policies.hgs.HGSConfig.crossover_rate
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.crossover_rate
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.policies.hgs.HGSConfig.local_search_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.local_search_iterations
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

````{py:attribute} use_cross_exchange
:canonical: src.configs.policies.hgs.HGSConfig.use_cross_exchange
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.use_cross_exchange
```

````

````{py:attribute} use_lambda_interchange
:canonical: src.configs.policies.hgs.HGSConfig.use_lambda_interchange
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.use_lambda_interchange
```

````

````{py:attribute} lambda_max
:canonical: src.configs.policies.hgs.HGSConfig.lambda_max
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.lambda_max
```

````

````{py:attribute} use_ejection_chains
:canonical: src.configs.policies.hgs.HGSConfig.use_ejection_chains
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.use_ejection_chains
```

````

````{py:attribute} use_3opt
:canonical: src.configs.policies.hgs.HGSConfig.use_3opt
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.use_3opt
```

````

````{py:attribute} initial_penalty_capacity
:canonical: src.configs.policies.hgs.HGSConfig.initial_penalty_capacity
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.initial_penalty_capacity
```

````

````{py:attribute} penalty_increase
:canonical: src.configs.policies.hgs.HGSConfig.penalty_increase
:type: float
:value: >
   1.2

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.penalty_increase
```

````

````{py:attribute} penalty_decrease
:canonical: src.configs.policies.hgs.HGSConfig.penalty_decrease
:type: float
:value: >
   0.85

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.penalty_decrease
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

````{py:attribute} vrpp
:canonical: src.configs.policies.hgs.HGSConfig.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.configs.policies.hgs.HGSConfig.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.profit_aware_operators
```

````

````{py:attribute} must_go
:canonical: src.configs.policies.hgs.HGSConfig.must_go
:type: typing.Optional[typing.List[src.configs.policies.other.must_go.MustGoConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.must_go
```

````

````{py:attribute} post_processing
:canonical: src.configs.policies.hgs.HGSConfig.post_processing
:type: typing.Optional[typing.List[src.configs.policies.other.post_processing.PostProcessingConfig]]
:value: >
   None

```{autodoc2-docstring} src.configs.policies.hgs.HGSConfig.post_processing
```

````

`````
