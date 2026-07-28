# {py:mod}`src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split`

```{py:module} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_daily_loads <src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split.compute_daily_loads>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split.compute_daily_loads
    :summary:
    ```
* - {py:obj}`split_day <src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split.split_day>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split.split_day
    :summary:
    ```
* - {py:obj}`_split_unconstrained <src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split._split_unconstrained>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split._split_unconstrained
    :summary:
    ```
* - {py:obj}`_split_constrained <src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split._split_constrained>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split._split_constrained
    :summary:
    ```
````

### API

````{py:function} compute_daily_loads(patterns: numpy.ndarray, base_wastes: numpy.ndarray, daily_increments: numpy.ndarray, T: int) -> numpy.ndarray
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split.compute_daily_loads

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split.compute_daily_loads
```
````

````{py:function} split_day(giant_tour: numpy.ndarray, loads: numpy.ndarray, distance_matrix: numpy.ndarray, capacity: float, n_vehicles: int) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split.split_day

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split.split_day
```
````

````{py:function} _split_unconstrained(giant_tour: numpy.ndarray, loads: numpy.ndarray, dist: numpy.ndarray, capacity: float) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split._split_unconstrained

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split._split_unconstrained
```
````

````{py:function} _split_constrained(giant_tour: numpy.ndarray, loads: numpy.ndarray, dist: numpy.ndarray, capacity: float, K: int) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split._split_constrained

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split._split_constrained
```
````
