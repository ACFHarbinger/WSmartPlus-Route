# {py:mod}`src.policies.hybrid_genetic_search_ruin_and_recreate.params`

```{py:module} src.policies.hybrid_genetic_search_ruin_and_recreate.params
```

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSRRParams <src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams>`
  - ```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams
    :summary:
    ```
````

### API

`````{py:class} HGSRRParams
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams
```

````{py:attribute} time_limit
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.time_limit
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.time_limit
```

````

````{py:attribute} population_size
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.population_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.population_size
```

````

````{py:attribute} elite_size
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.elite_size
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.elite_size
```

````

````{py:attribute} mutation_rate
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.mutation_rate
:type: float
:value: >
   0.3

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.mutation_rate
```

````

````{py:attribute} n_iterations_no_improvement
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.n_iterations_no_improvement
:type: int
:value: >
   20000

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.n_iterations_no_improvement
```

````

````{py:attribute} alpha_diversity
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.alpha_diversity
:type: float
:value: >
   0.5

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.alpha_diversity
```

````

````{py:attribute} min_diversity
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.min_diversity
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.min_diversity
```

````

````{py:attribute} diversity_change_rate
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.diversity_change_rate
:type: float
:value: >
   0.05

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.diversity_change_rate
```

````

````{py:attribute} no_improvement_threshold
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.no_improvement_threshold
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.no_improvement_threshold
```

````

````{py:attribute} survivor_threshold
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.survivor_threshold
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.survivor_threshold
```

````

````{py:attribute} max_vehicles
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.max_vehicles
:type: int
:value: >
   0

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.max_vehicles
```

````

````{py:attribute} crossover_rate
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.crossover_rate
:type: float
:value: >
   0.7

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.crossover_rate
```

````

````{py:attribute} neighbor_list_size
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.neighbor_list_size
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.neighbor_list_size
```

````

````{py:attribute} min_removal_pct
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.min_removal_pct
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.min_removal_pct
```

````

````{py:attribute} max_removal_pct
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.max_removal_pct
:type: float
:value: >
   0.4

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.max_removal_pct
```

````

````{py:attribute} noise_factor
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.noise_factor
:type: float
:value: >
   0.015

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.noise_factor
```

````

````{py:attribute} reaction_factor
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.reaction_factor
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.reaction_factor
```

````

````{py:attribute} decay_parameter
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.decay_parameter
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.decay_parameter
```

````

````{py:attribute} destroy_operators
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.destroy_operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.destroy_operators
```

````

````{py:attribute} repair_operators
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.repair_operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.repair_operators
```

````

````{py:attribute} operator_decay_rate
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.operator_decay_rate
:type: float
:value: >
   0.95

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.operator_decay_rate
```

````

````{py:attribute} score_sigma_1
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.score_sigma_1
:type: float
:value: >
   33.0

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.score_sigma_1
```

````

````{py:attribute} score_sigma_2
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.score_sigma_2
:type: float
:value: >
   9.0

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.score_sigma_2
```

````

````{py:attribute} score_sigma_3
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.score_sigma_3
:type: float
:value: >
   3.0

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.score_sigma_3
```

````

````{py:attribute} seed
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.seed
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.vrpp
```

````

````{py:attribute} profit_aware_operators
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.profit_aware_operators
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.profit_aware_operators
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams.from_config
```

````

`````
