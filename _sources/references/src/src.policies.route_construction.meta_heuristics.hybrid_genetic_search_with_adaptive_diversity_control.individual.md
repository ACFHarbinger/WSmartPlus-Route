# {py:mod}`src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual`

```{py:module} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Individual <src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual
    :summary:
    ```
````

### API

`````{py:class} Individual
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual
```

````{py:attribute} patterns
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.patterns
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.patterns
```

````

````{py:attribute} giant_tours
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.giant_tours
:type: typing.List[numpy.ndarray]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.giant_tours
```

````

````{py:attribute} cost
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.cost
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.cost
```

````

````{py:attribute} capacity_violations
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.capacity_violations
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.capacity_violations
```

````

````{py:attribute} fit
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.fit
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.fit
```

````

````{py:attribute} dc
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.dc
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.dc
```

````

````{py:attribute} biased_fitness
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.biased_fitness
:type: float
:value: >
   0.0

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.biased_fitness
```

````

````{py:attribute} is_feasible
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.is_feasible
:type: bool
:value: >
   False

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.is_feasible
```

````

````{py:attribute} routes
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.routes
:type: typing.List[typing.List[typing.List[int]]]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.routes
```

````

````{py:method} is_active(node: int, day_t: int) -> bool
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.is_active

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.is_active
```

````

````{py:method} set_active(node: int, day_t: int, active: bool) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.set_active

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual.set_active
```

````

`````
