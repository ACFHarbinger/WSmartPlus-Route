# {py:mod}`src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome`

```{py:module} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome
```

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Chromosome <src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome>`
  - ```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_adaptive_thresholds <src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.compute_adaptive_thresholds>`
  - ```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.compute_adaptive_thresholds
    :summary:
    ```
````

### API

````{py:function} compute_adaptive_thresholds(overflow_risk: numpy.ndarray, threshold_min: float = 0.1, threshold_max: float = 0.9) -> numpy.ndarray
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.compute_adaptive_thresholds

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.compute_adaptive_thresholds
```
````

`````{py:class} Chromosome(keys: numpy.ndarray, n_bins: int)
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome.__init__
```

````{py:attribute} __slots__
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome.__slots__
:value: >
   ('keys', 'n_bins')

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome.__slots__
```

````

````{py:method} random(n_bins: int, rng: numpy.random.Generator) -> src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome.random
:classmethod:

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome.random
```

````

````{py:method} from_selection_and_order(n_bins: int, selected_bins_0idx: typing.List[int], routing_order_0idx: typing.List[int], overhead: float = 0.15, rng: typing.Optional[numpy.random.Generator] = None) -> src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome.from_selection_and_order
:classmethod:

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome.from_selection_and_order
```

````

````{py:method} decode_selection(thresholds: numpy.ndarray) -> typing.List[int]
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome.decode_selection

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome.decode_selection
```

````

````{py:method} decode_routing_order(selected_1based: typing.List[int]) -> typing.List[int]
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome.decode_routing_order

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome.decode_routing_order
```

````

````{py:method} to_routes(thresholds: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, mandatory_override: typing.Optional[typing.List[int]] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome.to_routes

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome.to_routes
```

````

`````
