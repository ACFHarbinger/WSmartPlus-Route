# {py:mod}`src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two`

```{py:module} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two
```

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TourPopulation <src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.TourPopulation>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.TourPopulation
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ipt_crossover <src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.ipt_crossover>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.ipt_crossover
    :summary:
    ```
* - {py:obj}`solve_lkh2 <src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.solve_lkh2>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.solve_lkh2
    :summary:
    ```
* - {py:obj}`solve_lkh <src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.solve_lkh>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.solve_lkh
    :summary:
    ```
````

### API

`````{py:class} TourPopulation(max_size: int = 10, diversity_threshold: float = 0.05)
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.TourPopulation

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.TourPopulation
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.TourPopulation.__init__
```

````{py:method} _to_edge_set(tour: typing.List[int]) -> typing.Set[typing.Tuple[int, int]]
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.TourPopulation._to_edge_set
:staticmethod:

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.TourPopulation._to_edge_set
```

````

````{py:method} _hamming_distance(es_new: typing.Set[typing.Tuple[int, int]], n: int) -> float
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.TourPopulation._hamming_distance

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.TourPopulation._hamming_distance
```

````

````{py:method} try_insert(tour: typing.List[int], cost: float) -> bool
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.TourPopulation.try_insert

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.TourPopulation.try_insert
```

````

````{py:method} sample_parents(rng: numpy.random.Generator, n_parents: int = 2) -> typing.List[typing.Tuple[typing.List[int], float]]
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.TourPopulation.sample_parents

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.TourPopulation.sample_parents
```

````

````{py:property} size
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.TourPopulation.size
:type: int

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.TourPopulation.size
```

````

````{py:property} diversity
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.TourPopulation.diversity
:type: float

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.TourPopulation.diversity
```

````

`````

````{py:function} ipt_crossover(parent1: typing.List[int], parent2: typing.List[int], distance_matrix: numpy.ndarray, candidates: typing.Dict[int, typing.List[int]], stdlib_rng: random.Random, max_k: int = 5, max_gap_iter: int = 20) -> typing.Tuple[typing.List[int], float]
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.ipt_crossover

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.ipt_crossover
```
````

````{py:function} solve_lkh2(distance_matrix: numpy.ndarray, initial_tour: typing.Optional[typing.List[int]] = None, max_iterations: int = 200, max_k: int = 5, n_candidates: int = 5, population_size: int = 10, diversity_threshold: float = 0.05, crossover_prob: float = 0.6, sg_max_iter: int = 100, sg_mu_init: float = 1.0, sg_patience: int = 20, local_search_iter_per_offspring: int = 50, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None, np_rng: typing.Optional[numpy.random.Generator] = None, seed: typing.Optional[int] = None) -> typing.Tuple[typing.List[int], float, float]
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.solve_lkh2

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.solve_lkh2
```
````

````{py:function} solve_lkh(distance_matrix: numpy.ndarray, initial_tour: typing.Optional[typing.List[int]] = None, max_iterations: int = 200, max_k: int = 5, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None, np_rng: typing.Optional[numpy.random.Generator] = None, seed: typing.Optional[int] = None) -> typing.Tuple[typing.List[int], float]
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.solve_lkh

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two.solve_lkh
```
````
