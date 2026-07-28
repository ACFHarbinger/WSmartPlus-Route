# {py:mod}`src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.search.deterministic`

```{py:module} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.search.deterministic
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.search.deterministic
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_evaluate_move <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.search.deterministic._evaluate_move>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.search.deterministic._evaluate_move
    :summary:
    ```
* - {py:obj}`local_search_2 <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.search.deterministic.local_search_2>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.search.deterministic.local_search_2
    :summary:
    ```
````

### API

````{py:function} _evaluate_move(routes_list: typing.List[typing.List[int]], idx_route: int, position: int, bin_to_move: int, previous_solution: typing.List[typing.List[int]], previous_profit: float, p_vehicle: float, p_load: float, p_route_difference: float, p_shift: float, data: dict, distance_matrix: numpy.ndarray, values: numpy.ndarray) -> typing.Tuple[float, bool]
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.search.deterministic._evaluate_move

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.search.deterministic._evaluate_move
```
````

````{py:function} local_search_2(previous_solution: typing.List[typing.List[int]], p_vehicle: float, p_load: float, p_route_difference: float, p_shift: float, data: dict, distance_matrix: numpy.ndarray, values: numpy.ndarray) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.search.deterministic.local_search_2

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.search.deterministic.local_search_2
```
````
