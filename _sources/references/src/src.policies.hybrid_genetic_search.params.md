# {py:mod}`src.policies.hybrid_genetic_search.params`

```{py:module} src.policies.hybrid_genetic_search.params
```

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSParams <src.policies.hybrid_genetic_search.params.HGSParams>`
  - ```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams
    :summary:
    ```
````

### API

`````{py:class} HGSParams
:canonical: src.policies.hybrid_genetic_search.params.HGSParams

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams
```

````{py:attribute} time_limit
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.time_limit
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.time_limit
```

````

````{py:attribute} mu
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.mu
:type: int
:value: >
   25

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.mu
```

````

````{py:attribute} n_offspring
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.n_offspring
:type: int
:value: >
   40

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.n_offspring
```

````

````{py:attribute} nb_elite
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.nb_elite
:type: int
:value: >
   4

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.nb_elite
```

````

````{py:attribute} nb_close
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.nb_close
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.nb_close
```

````

````{py:attribute} nb_granular
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.nb_granular
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.nb_granular
```

````

````{py:attribute} target_feasible
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.target_feasible
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.target_feasible
```

````

````{py:attribute} n_iterations_no_improvement
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.n_iterations_no_improvement
:type: int
:value: >
   20000

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.n_iterations_no_improvement
```

````

````{py:attribute} mutation_rate
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.mutation_rate
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.mutation_rate
```

````

````{py:attribute} repair_probability
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.repair_probability
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.repair_probability
```

````

````{py:attribute} crossover_rate
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.crossover_rate
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.crossover_rate
```

````

````{py:attribute} alpha_diversity
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.alpha_diversity
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.alpha_diversity
```

````

````{py:attribute} min_diversity
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.min_diversity
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.min_diversity
```

````

````{py:attribute} diversity_change_rate
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.diversity_change_rate
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.diversity_change_rate
```

````

````{py:attribute} local_search_iterations
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.local_search_iterations
:type: int
:value: >
   500

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.local_search_iterations
```

````

````{py:attribute} max_vehicles
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.max_vehicles
:type: int
:value: >
   0

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.max_vehicles
```

````

````{py:attribute} initial_penalty_capacity
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.initial_penalty_capacity
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.initial_penalty_capacity
```

````

````{py:attribute} penalty_increase
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.penalty_increase
:type: float
:value: >
   1.2

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.penalty_increase
```

````

````{py:attribute} penalty_decrease
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.penalty_decrease
:type: float
:value: >
   0.85

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.penalty_decrease
```

````

````{py:attribute} seed
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.profit_aware_operators
```

````

````{py:method} from_config(config: logic.src.configs.policies.HGSConfig) -> src.policies.hybrid_genetic_search.params.HGSParams
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.from_config
```

````

````{py:property} lambda_param
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.lambda_param
:type: int

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.lambda_param
```

````

````{py:property} population_size
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.population_size
:type: int

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.population_size
```

````

````{py:property} elite_size
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.elite_size
:type: int

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.elite_size
```

````

````{py:property} min_diversity_threshold
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.min_diversity_threshold
:type: float

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.min_diversity_threshold
```

````

````{py:property} no_improvement_threshold
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.no_improvement_threshold
:type: int

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.no_improvement_threshold
```

````

````{py:property} neighbor_list_size
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.neighbor_list_size
:type: int

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.neighbor_list_size
```

````

`````
