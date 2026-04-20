# {py:mod}`src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.objectives`

```{py:module} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.objectives
```

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.objectives
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_overflow_risk <src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.objectives.compute_overflow_risk>`
  - ```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.objectives.compute_overflow_risk
    :summary:
    ```
* - {py:obj}`evaluate_chromosome <src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.objectives.evaluate_chromosome>`
  - ```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.objectives.evaluate_chromosome
    :summary:
    ```
* - {py:obj}`evaluate_population <src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.objectives.evaluate_population>`
  - ```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.objectives.evaluate_population
    :summary:
    ```
````

### API

````{py:function} compute_overflow_risk(current_fill: numpy.ndarray, bin_mass: numpy.ndarray, scenario_tree: typing.Optional[typing.Any], overflow_penalty_frac: float) -> numpy.ndarray
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.objectives.compute_overflow_risk

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.objectives.compute_overflow_risk
```
````

````{py:function} evaluate_chromosome(chrom: logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome, thresholds: numpy.ndarray, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, overflow_risk: numpy.ndarray, overflow_penalty: float, mandatory_override: typing.Optional[typing.List[int]] = None) -> typing.Tuple[float, float, float]
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.objectives.evaluate_chromosome

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.objectives.evaluate_chromosome
```
````

````{py:function} evaluate_population(population: typing.List[logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome], thresholds: numpy.ndarray, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, overflow_risk: numpy.ndarray, overflow_penalty: float, mandatory_override: typing.Optional[typing.List[int]] = None) -> numpy.ndarray
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.objectives.evaluate_population

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.objectives.evaluate_population
```
````
