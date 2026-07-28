# {py:mod}`src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution`

```{py:module} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_extract_edges <src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution._extract_edges>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution._extract_edges
    :summary:
    ```
* - {py:obj}`_compute_broken_pairs_distance <src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution._compute_broken_pairs_distance>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution._compute_broken_pairs_distance
    :summary:
    ```
* - {py:obj}`update_biased_fitness <src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution.update_biased_fitness>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution.update_biased_fitness
    :summary:
    ```
* - {py:obj}`evaluate <src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution.evaluate>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution.evaluate
    :summary:
    ```
````

### API

````{py:function} _extract_edges(individual: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual) -> set
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution._extract_edges

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution._extract_edges
```
````

````{py:function} _compute_broken_pairs_distance(ind1: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual, ind2: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual) -> float
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution._compute_broken_pairs_distance

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution._compute_broken_pairs_distance
```
````

````{py:function} update_biased_fitness(population: typing.List[src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual], nb_elite: int, neighbor_size: int = 5, distance_cache: typing.Optional[dict] = None, inv: typing.Optional[dict] = None)
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution.update_biased_fitness

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution.update_biased_fitness
```
````

````{py:function} evaluate(ind: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual, split_manager: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.split.LinearSplit, penalty_capacity: float = 1.0)
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution.evaluate

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution.evaluate
```
````
