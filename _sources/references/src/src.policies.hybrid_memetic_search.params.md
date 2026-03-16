# {py:mod}`src.policies.hybrid_memetic_search.params`

```{py:module} src.policies.hybrid_memetic_search.params
```

```{autodoc2-docstring} src.policies.hybrid_memetic_search.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HybridMemeticSearchParams <src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams>`
  - ```{autodoc2-docstring} src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams
    :summary:
    ```
````

### API

`````{py:class} HybridMemeticSearchParams
:canonical: src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams

```{autodoc2-docstring} src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams
```

````{py:attribute} population_size
:canonical: src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.population_size
:type: int
:value: >
   30

```{autodoc2-docstring} src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.population_size
```

````

````{py:attribute} max_generations
:canonical: src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.max_generations
:type: int
:value: >
   100

```{autodoc2-docstring} src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.max_generations
```

````

````{py:attribute} substitution_rate
:canonical: src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.substitution_rate
:type: float
:value: >
   0.2

```{autodoc2-docstring} src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.substitution_rate
```

````

````{py:attribute} crossover_rate
:canonical: src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.crossover_rate
:type: float
:value: >
   0.8

```{autodoc2-docstring} src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.crossover_rate
```

````

````{py:attribute} mutation_rate
:canonical: src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.mutation_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.mutation_rate
```

````

````{py:attribute} elitism_count
:canonical: src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.elitism_count
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.elitism_count
```

````

````{py:attribute} aco_init_iterations
:canonical: src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.aco_init_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.aco_init_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.time_limit
:type: float
:value: >
   300.0

```{autodoc2-docstring} src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.time_limit
```

````

````{py:attribute} aco_params
:canonical: src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.aco_params
:type: src.policies.ant_colony_optimization_k_sparse.params.ACOParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.aco_params
```

````

````{py:attribute} alns_params
:canonical: src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.alns_params
:type: src.policies.adaptive_large_neighborhood_search.ALNSParams
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.alns_params
```

````

````{py:method} __post_init__()
:canonical: src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.__post_init__

```{autodoc2-docstring} src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.__post_init__
```

````

````{py:method} from_config(config: typing.Any) -> src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams
:canonical: src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.from_config
```

````

````{py:property} max_iterations
:canonical: src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.max_iterations
:type: int

```{autodoc2-docstring} src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams.max_iterations
```

````

`````
