# {py:mod}`src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods`

```{py:module} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods
```

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_neighbors <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.get_neighbors>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.get_neighbors
    :summary:
    ```
* - {py:obj}`get_2opt_neighbors <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.get_2opt_neighbors>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.get_2opt_neighbors
    :summary:
    ```
* - {py:obj}`relocate_within_route <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.relocate_within_route>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.relocate_within_route
    :summary:
    ```
* - {py:obj}`cross_exchange <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.cross_exchange>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.cross_exchange
    :summary:
    ```
* - {py:obj}`or_opt_move <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.or_opt_move>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.or_opt_move
    :summary:
    ```
* - {py:obj}`move_between_routes <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.move_between_routes>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.move_between_routes
    :summary:
    ```
* - {py:obj}`insert_bin_in_route <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.insert_bin_in_route>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.insert_bin_in_route
    :summary:
    ```
* - {py:obj}`mutate_route_by_swapping_bins <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.mutate_route_by_swapping_bins>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.mutate_route_by_swapping_bins
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.__all__>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.__all__
:value: >
   ['get_neighbors', 'get_2opt_neighbors', 'relocate_within_route', 'cross_exchange', 'or_opt_move', 'm...

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.__all__
```

````

````{py:function} get_neighbors(route: list) -> list
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.get_neighbors

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.get_neighbors
```
````

````{py:function} get_2opt_neighbors(route: list) -> list
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.get_2opt_neighbors

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.get_2opt_neighbors
```
````

````{py:function} relocate_within_route(route: list) -> list
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.relocate_within_route

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.relocate_within_route
```
````

````{py:function} cross_exchange(routes: list) -> list
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.cross_exchange

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.cross_exchange
```
````

````{py:function} or_opt_move(route: list) -> list
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.or_opt_move

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.or_opt_move
```
````

````{py:function} move_between_routes(routes: list, data, vehicle_capacity: float, id_to_index: dict) -> list
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.move_between_routes

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.move_between_routes
```
````

````{py:function} insert_bin_in_route(route: list, bin_id: int, id_to_index: dict, distance_matrix) -> list
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.insert_bin_in_route

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.insert_bin_in_route
```
````

````{py:function} mutate_route_by_swapping_bins(route: list, num_bins: int = 1) -> list
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.mutate_route_by_swapping_bins

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_neighborhoods.mutate_route_by_swapping_bins
```
````
