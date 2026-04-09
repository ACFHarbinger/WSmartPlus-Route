# {py:mod}`src.configs.rl.policies.hgs`

```{py:module} src.configs.rl.policies.hgs
```

```{autodoc2-docstring} src.configs.rl.policies.hgs
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSConfig <src.configs.rl.policies.hgs.HGSConfig>`
  - ```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig
    :summary:
    ```
````

### API

`````{py:class} HGSConfig
:canonical: src.configs.rl.policies.hgs.HGSConfig

```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig
```

````{py:attribute} time_limit
:canonical: src.configs.rl.policies.hgs.HGSConfig.time_limit
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig.time_limit
```

````

````{py:attribute} mu
:canonical: src.configs.rl.policies.hgs.HGSConfig.mu
:type: int
:value: >
   25

```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig.mu
```

````

````{py:attribute} lambda_param
:canonical: src.configs.rl.policies.hgs.HGSConfig.lambda_param
:type: int
:value: >
   40

```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig.lambda_param
```

````

````{py:attribute} nb_elite
:canonical: src.configs.rl.policies.hgs.HGSConfig.nb_elite
:type: int
:value: >
   4

```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig.nb_elite
```

````

````{py:attribute} nb_close
:canonical: src.configs.rl.policies.hgs.HGSConfig.nb_close
:type: int
:value: >
   5

```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig.nb_close
```

````

````{py:attribute} nb_granular
:canonical: src.configs.rl.policies.hgs.HGSConfig.nb_granular
:type: int
:value: >
   20

```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig.nb_granular
```

````

````{py:attribute} target_feasible
:canonical: src.configs.rl.policies.hgs.HGSConfig.target_feasible
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig.target_feasible
```

````

````{py:attribute} n_iterations_no_improvement
:canonical: src.configs.rl.policies.hgs.HGSConfig.n_iterations_no_improvement
:type: int
:value: >
   20000

```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig.n_iterations_no_improvement
```

````

````{py:attribute} mutation_rate
:canonical: src.configs.rl.policies.hgs.HGSConfig.mutation_rate
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig.mutation_rate
```

````

````{py:attribute} repair_probability
:canonical: src.configs.rl.policies.hgs.HGSConfig.repair_probability
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig.repair_probability
```

````

````{py:attribute} crossover_rate
:canonical: src.configs.rl.policies.hgs.HGSConfig.crossover_rate
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig.crossover_rate
```

````

````{py:attribute} min_diversity
:canonical: src.configs.rl.policies.hgs.HGSConfig.min_diversity
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig.min_diversity
```

````

````{py:attribute} diversity_change_rate
:canonical: src.configs.rl.policies.hgs.HGSConfig.diversity_change_rate
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig.diversity_change_rate
```

````

````{py:attribute} local_search_iterations
:canonical: src.configs.rl.policies.hgs.HGSConfig.local_search_iterations
:type: int
:value: >
   100

```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig.local_search_iterations
```

````

````{py:attribute} max_vehicles
:canonical: src.configs.rl.policies.hgs.HGSConfig.max_vehicles
:type: int
:value: >
   0

```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig.max_vehicles
```

````

````{py:attribute} initial_penalty_capacity
:canonical: src.configs.rl.policies.hgs.HGSConfig.initial_penalty_capacity
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig.initial_penalty_capacity
```

````

````{py:attribute} penalty_increase
:canonical: src.configs.rl.policies.hgs.HGSConfig.penalty_increase
:type: float
:value: >
   1.2

```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig.penalty_increase
```

````

````{py:attribute} penalty_decrease
:canonical: src.configs.rl.policies.hgs.HGSConfig.penalty_decrease
:type: float
:value: >
   0.85

```{autodoc2-docstring} src.configs.rl.policies.hgs.HGSConfig.penalty_decrease
```

````

`````
