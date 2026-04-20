# {py:mod}`src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.population`

```{py:module} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.population
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.population
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Population <src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.population.Population>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.population.Population
    :summary:
    ```
````

### API

`````{py:class} Population(target_size: int, nb_close: int)
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.population.Population

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.population.Population
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.population.Population.__init__
```

````{py:method} add_individual(ind: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.population.Population.add_individual

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.population.Population.add_individual
```

````

````{py:method} compute_diversity(subpop: typing.List[src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual], T: int) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.population.Population.compute_diversity

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.population.Population.compute_diversity
```

````

````{py:method} _distance(ind_a: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual, ind_b: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual, T: int) -> float
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.population.Population._distance

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.population.Population._distance
```

````

````{py:method} rank_and_survive(subpop: typing.List[src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual], T: int) -> typing.List[src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.population.Population.rank_and_survive

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.population.Population.rank_and_survive
```

````

````{py:method} trigger_survivor_selection(T: int) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.population.Population.trigger_survivor_selection

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.population.Population.trigger_survivor_selection
```

````

`````
