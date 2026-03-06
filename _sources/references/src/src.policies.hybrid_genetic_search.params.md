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
   60.0

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.time_limit
```

````

````{py:attribute} population_size
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.population_size
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.population_size
```

````

````{py:attribute} elite_size
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.elite_size
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.elite_size
```

````

````{py:attribute} mutation_rate
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.mutation_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.mutation_rate
```

````

````{py:attribute} crossover_rate
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.crossover_rate
:type: float
:value: >
   0.7

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.crossover_rate
```

````

````{py:attribute} n_generations
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.n_generations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.n_generations
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

````{py:attribute} survivor_threshold
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.survivor_threshold
:type: int
:value: >
   2

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.survivor_threshold
```

````

````{py:attribute} no_improvement_threshold
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.no_improvement_threshold
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.no_improvement_threshold
```

````

````{py:attribute} min_diversity_threshold
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.min_diversity_threshold
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.min_diversity_threshold
```

````

````{py:attribute} neighbor_list_size
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.neighbor_list_size
:type: int
:value: >
   15

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.neighbor_list_size
```

````

````{py:attribute} local_search_iterations
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.local_search_iterations
:type: int
:value: >
   100

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

````{py:method} from_config(config: logic.src.configs.policies.HGSConfig) -> src.policies.hybrid_genetic_search.params.HGSParams
:canonical: src.policies.hybrid_genetic_search.params.HGSParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.hybrid_genetic_search.params.HGSParams.from_config
```

````

`````
