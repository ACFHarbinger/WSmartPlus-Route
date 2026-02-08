# {py:mod}`src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations`

```{py:module} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations
```

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`remove_bins_from_route <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.remove_bins_from_route>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.remove_bins_from_route
    :summary:
    ```
* - {py:obj}`move_n_route_random <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.move_n_route_random>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.move_n_route_random
    :summary:
    ```
* - {py:obj}`swap_n_route_random <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.swap_n_route_random>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.swap_n_route_random
    :summary:
    ```
* - {py:obj}`remove_n_bins_random <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.remove_n_bins_random>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.remove_n_bins_random
    :summary:
    ```
* - {py:obj}`add_n_bins_random <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.add_n_bins_random>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.add_n_bins_random
    :summary:
    ```
* - {py:obj}`add_route_with_removed_bins_random <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.add_route_with_removed_bins_random>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.add_route_with_removed_bins_random
    :summary:
    ```
* - {py:obj}`move_n_route_consecutive <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.move_n_route_consecutive>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.move_n_route_consecutive
    :summary:
    ```
* - {py:obj}`swap_n_route_consecutive <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.swap_n_route_consecutive>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.swap_n_route_consecutive
    :summary:
    ```
* - {py:obj}`remove_n_bins_consecutive <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.remove_n_bins_consecutive>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.remove_n_bins_consecutive
    :summary:
    ```
* - {py:obj}`add_n_bins_consecutive <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.add_n_bins_consecutive>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.add_n_bins_consecutive
    :summary:
    ```
* - {py:obj}`add_route_with_removed_bins_consecutive <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.add_route_with_removed_bins_consecutive>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.add_route_with_removed_bins_consecutive
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.__all__>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.__all__
:value: >
   ['remove_bins_from_route', 'move_n_route_random', 'swap_n_route_random', 'remove_n_bins_random', 'ad...

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.__all__
```

````

````{py:function} remove_bins_from_route(route: list, num_bins: int = 1) -> list
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.remove_bins_from_route

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.remove_bins_from_route
```
````

````{py:function} move_n_route_random(routes_list: list, n: int = 2) -> list
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.move_n_route_random

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.move_n_route_random
```
````

````{py:function} swap_n_route_random(routes_list: list, n: int = 2) -> list
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.swap_n_route_random

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.swap_n_route_random
```
````

````{py:function} remove_n_bins_random(routes_list: list, removed_bins: set, bins_cannot_removed: set, n: int = 2) -> list
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.remove_n_bins_random

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.remove_n_bins_random
```
````

````{py:function} add_n_bins_random(routes_list: list, removed_bins: set, stocks: dict, vehicle_capacity: float, id_to_index: dict, distance_matrix, n: int = 2) -> list
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.add_n_bins_random

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.add_n_bins_random
```
````

````{py:function} add_route_with_removed_bins_random(routes_list: list, removed_bins: set, stocks: dict, vehicle_capacity: float) -> list
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.add_route_with_removed_bins_random

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.add_route_with_removed_bins_random
```
````

````{py:function} move_n_route_consecutive(routes_list: list, n: int = 2) -> list
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.move_n_route_consecutive

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.move_n_route_consecutive
```
````

````{py:function} swap_n_route_consecutive(routes_list: list, n: int = 2) -> list
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.swap_n_route_consecutive

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.swap_n_route_consecutive
```
````

````{py:function} remove_n_bins_consecutive(routes_list: list, removed_bins: set, bins_cannot_removed: set, n: int = 2) -> list
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.remove_n_bins_consecutive

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.remove_n_bins_consecutive
```
````

````{py:function} add_n_bins_consecutive(routes_list: list, removed_bins: set, stocks: dict, vehicle_capacity: float, id_to_index: dict, distance_matrix, n: int = 2) -> list
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.add_n_bins_consecutive

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.add_n_bins_consecutive
```
````

````{py:function} add_route_with_removed_bins_consecutive(routes_list: list, removed_bins: set, stocks: dict, vehicle_capacity: float) -> list
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.add_route_with_removed_bins_consecutive

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans_perturbations.add_route_with_removed_bins_consecutive
```
````
