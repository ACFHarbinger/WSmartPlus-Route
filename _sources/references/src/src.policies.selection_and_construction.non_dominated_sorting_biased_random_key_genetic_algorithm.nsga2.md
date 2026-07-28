# {py:mod}`src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.nsga2`

```{py:module} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.nsga2
```

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.nsga2
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`fast_non_dominated_sort <src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.nsga2.fast_non_dominated_sort>`
  - ```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.nsga2.fast_non_dominated_sort
    :summary:
    ```
* - {py:obj}`crowding_distance <src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.nsga2.crowding_distance>`
  - ```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.nsga2.crowding_distance
    :summary:
    ```
* - {py:obj}`select_elite_nsga2 <src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.nsga2.select_elite_nsga2>`
  - ```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.nsga2.select_elite_nsga2
    :summary:
    ```
````

### API

````{py:function} fast_non_dominated_sort(objectives: numpy.ndarray) -> typing.List[typing.List[int]]
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.nsga2.fast_non_dominated_sort

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.nsga2.fast_non_dominated_sort
```
````

````{py:function} crowding_distance(front: typing.List[int], objectives: numpy.ndarray) -> numpy.ndarray
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.nsga2.crowding_distance

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.nsga2.crowding_distance
```
````

````{py:function} select_elite_nsga2(objectives: numpy.ndarray, n_elite: int) -> typing.Tuple[typing.List[int], numpy.ndarray]
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.nsga2.select_elite_nsga2

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.nsga2.select_elite_nsga2
```
````
