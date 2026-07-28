# {py:mod}`src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params`

```{py:module} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params
```

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NDSBRKGAParams <src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams>`
  - ```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams
    :summary:
    ```
````

### API

`````{py:class} NDSBRKGAParams
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams
```

````{py:attribute} pop_size
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.pop_size
:type: int
:value: >
   60

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.pop_size
```

````

````{py:attribute} n_elite
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.n_elite
:type: int
:value: >
   15

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.n_elite
```

````

````{py:attribute} n_mutants
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.n_mutants
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.n_mutants
```

````

````{py:attribute} bias_elite
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.bias_elite
:type: float
:value: >
   0.7

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.bias_elite
```

````

````{py:attribute} max_generations
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.max_generations
:type: int
:value: >
   200

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.max_generations
```

````

````{py:attribute} time_limit
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.time_limit
:type: float
:value: >
   90.0

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.time_limit
```

````

````{py:attribute} seed
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.seed
:type: typing.Optional[int]
:value: >
   42

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.seed
```

````

````{py:attribute} vrpp
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.vrpp
:type: bool
:value: >
   True

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.vrpp
```

````

````{py:attribute} overflow_penalty
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.overflow_penalty
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.overflow_penalty
```

````

````{py:attribute} seed_selection_strategy
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.seed_selection_strategy
:type: str
:value: >
   'fractional_knapsack'

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.seed_selection_strategy
```

````

````{py:attribute} seed_routing_strategy
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.seed_routing_strategy
:type: str
:value: >
   'greedy'

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.seed_routing_strategy
```

````

````{py:attribute} n_seed_solutions
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.n_seed_solutions
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.n_seed_solutions
```

````

````{py:attribute} selection_threshold_min
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.selection_threshold_min
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.selection_threshold_min
```

````

````{py:attribute} selection_threshold_max
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.selection_threshold_max
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.selection_threshold_max
```

````

````{py:method} from_config(config: object) -> src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.from_config
:classmethod:

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams.from_config
```

````

`````
