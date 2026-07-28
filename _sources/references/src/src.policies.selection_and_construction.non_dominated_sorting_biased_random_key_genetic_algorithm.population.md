# {py:mod}`src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population`

```{py:module} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population
```

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Population <src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population.Population>`
  - ```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population.Population
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_seed_selection <src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population._seed_selection>`
  - ```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population._seed_selection
    :summary:
    ```
* - {py:obj}`_seed_routing_order <src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population._seed_routing_order>`
  - ```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population._seed_routing_order
    :summary:
    ```
* - {py:obj}`_build_seed_chromosomes <src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population._build_seed_chromosomes>`
  - ```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population._build_seed_chromosomes
    :summary:
    ```
````

### API

````{py:function} _seed_selection(n_bins: int, current_fill: numpy.ndarray, distance_matrix: numpy.ndarray, capacity: float, revenue_kg: float, bin_density: float, bin_volume: float, max_fill: float, overflow_penalty_frac: float, scenario_tree: typing.Optional[object], strategy_name: str, rng: numpy.random.Generator) -> typing.List[int]
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population._seed_selection

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population._seed_selection
```
````

````{py:function} _seed_routing_order(selected_1based: typing.List[int], dist_matrix: numpy.ndarray) -> typing.List[int]
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population._seed_routing_order

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population._seed_routing_order
```
````

````{py:function} _build_seed_chromosomes(n_bins: int, current_fill: numpy.ndarray, distance_matrix: numpy.ndarray, capacity: float, revenue_kg: float, bin_density: float, bin_volume: float, max_fill: float, overflow_penalty_frac: float, scenario_tree: typing.Optional[object], params: logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams, rng: numpy.random.Generator) -> typing.List[logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome]
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population._build_seed_chromosomes

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population._build_seed_chromosomes
```
````

`````{py:class} Population(chromosomes: typing.List[logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome], objectives: numpy.ndarray, front_ranks: numpy.ndarray)
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population.Population

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population.Population
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population.Population.__init__
```

````{py:method} initialise(n_bins: int, thresholds: numpy.ndarray, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, overflow_risk: numpy.ndarray, current_fill: numpy.ndarray, bin_density: float, bin_volume: float, max_fill: float, overflow_penalty_frac: float, scenario_tree: typing.Optional[object], params: logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.params.NDSBRKGAParams, mandatory_override: typing.Optional[typing.List[int]] = None) -> src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population.Population
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population.Population.initialise
:classmethod:

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population.Population.initialise
```

````

````{py:method} get_elite_indices(n_elite: int) -> typing.List[int]
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population.Population.get_elite_indices

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population.Population.get_elite_indices
```

````

````{py:method} breed_next_generation(n_elite: int, n_mutants: int, bias_elite: float, rng: numpy.random.Generator) -> typing.List[logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome]
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population.Population.breed_next_generation

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population.Population.breed_next_generation
```

````

````{py:method} best_chromosome() -> typing.Tuple[logic.src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.chromosome.Chromosome, numpy.ndarray]
:canonical: src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population.Population.best_chromosome

```{autodoc2-docstring} src.policies.selection_and_construction.non_dominated_sorting_biased_random_key_genetic_algorithm.population.Population.best_chromosome
```

````

`````
