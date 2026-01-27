# {py:mod}`src.policies.look_ahead_aux.sans_opt`

```{py:module} src.policies.look_ahead_aux.sans_opt
```

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_neighbors <src.policies.look_ahead_aux.sans_opt.get_neighbors>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.get_neighbors
    :summary:
    ```
* - {py:obj}`get_2opt_neighbors <src.policies.look_ahead_aux.sans_opt.get_2opt_neighbors>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.get_2opt_neighbors
    :summary:
    ```
* - {py:obj}`relocate_within_route <src.policies.look_ahead_aux.sans_opt.relocate_within_route>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.relocate_within_route
    :summary:
    ```
* - {py:obj}`cross_exchange <src.policies.look_ahead_aux.sans_opt.cross_exchange>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.cross_exchange
    :summary:
    ```
* - {py:obj}`or_opt_move <src.policies.look_ahead_aux.sans_opt.or_opt_move>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.or_opt_move
    :summary:
    ```
* - {py:obj}`move_between_routes <src.policies.look_ahead_aux.sans_opt.move_between_routes>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.move_between_routes
    :summary:
    ```
* - {py:obj}`insert_bin_in_route <src.policies.look_ahead_aux.sans_opt.insert_bin_in_route>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.insert_bin_in_route
    :summary:
    ```
* - {py:obj}`mutate_route_by_swapping_bins <src.policies.look_ahead_aux.sans_opt.mutate_route_by_swapping_bins>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.mutate_route_by_swapping_bins
    :summary:
    ```
* - {py:obj}`remove_bins_from_route <src.policies.look_ahead_aux.sans_opt.remove_bins_from_route>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.remove_bins_from_route
    :summary:
    ```
* - {py:obj}`move_n_route_random <src.policies.look_ahead_aux.sans_opt.move_n_route_random>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.move_n_route_random
    :summary:
    ```
* - {py:obj}`swap_n_route_random <src.policies.look_ahead_aux.sans_opt.swap_n_route_random>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.swap_n_route_random
    :summary:
    ```
* - {py:obj}`remove_n_bins_random <src.policies.look_ahead_aux.sans_opt.remove_n_bins_random>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.remove_n_bins_random
    :summary:
    ```
* - {py:obj}`add_n_bins_random <src.policies.look_ahead_aux.sans_opt.add_n_bins_random>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.add_n_bins_random
    :summary:
    ```
* - {py:obj}`add_route_with_removed_bins_random <src.policies.look_ahead_aux.sans_opt.add_route_with_removed_bins_random>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.add_route_with_removed_bins_random
    :summary:
    ```
* - {py:obj}`move_n_route_consecutive <src.policies.look_ahead_aux.sans_opt.move_n_route_consecutive>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.move_n_route_consecutive
    :summary:
    ```
* - {py:obj}`swap_n_route_consecutive <src.policies.look_ahead_aux.sans_opt.swap_n_route_consecutive>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.swap_n_route_consecutive
    :summary:
    ```
* - {py:obj}`remove_n_bins_consecutive <src.policies.look_ahead_aux.sans_opt.remove_n_bins_consecutive>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.remove_n_bins_consecutive
    :summary:
    ```
* - {py:obj}`add_n_bins_consecutive <src.policies.look_ahead_aux.sans_opt.add_n_bins_consecutive>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.add_n_bins_consecutive
    :summary:
    ```
* - {py:obj}`add_route_with_removed_bins_consecutive <src.policies.look_ahead_aux.sans_opt.add_route_with_removed_bins_consecutive>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.add_route_with_removed_bins_consecutive
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.policies.look_ahead_aux.sans_opt.__all__>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: src.policies.look_ahead_aux.sans_opt.__all__
:value: >
   ['get_neighbors', 'get_2opt_neighbors', 'relocate_within_route', 'cross_exchange', 'or_opt_move', 'm...

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.__all__
```

````

````{py:function} get_neighbors(route)
:canonical: src.policies.look_ahead_aux.sans_opt.get_neighbors

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.get_neighbors
```
````

````{py:function} get_2opt_neighbors(route)
:canonical: src.policies.look_ahead_aux.sans_opt.get_2opt_neighbors

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.get_2opt_neighbors
```
````

````{py:function} relocate_within_route(route)
:canonical: src.policies.look_ahead_aux.sans_opt.relocate_within_route

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.relocate_within_route
```
````

````{py:function} cross_exchange(routes)
:canonical: src.policies.look_ahead_aux.sans_opt.cross_exchange

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.cross_exchange
```
````

````{py:function} or_opt_move(route)
:canonical: src.policies.look_ahead_aux.sans_opt.or_opt_move

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.or_opt_move
```
````

````{py:function} move_between_routes(routes, data, vehicle_capacity, id_to_index)
:canonical: src.policies.look_ahead_aux.sans_opt.move_between_routes

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.move_between_routes
```
````

````{py:function} insert_bin_in_route(route, bin_id, id_to_index, distance_matrix)
:canonical: src.policies.look_ahead_aux.sans_opt.insert_bin_in_route

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.insert_bin_in_route
```
````

````{py:function} mutate_route_by_swapping_bins(route, num_bins=1)
:canonical: src.policies.look_ahead_aux.sans_opt.mutate_route_by_swapping_bins

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.mutate_route_by_swapping_bins
```
````

````{py:function} remove_bins_from_route(route, num_bins=1)
:canonical: src.policies.look_ahead_aux.sans_opt.remove_bins_from_route

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.remove_bins_from_route
```
````

````{py:function} move_n_route_random(routes_list, n=2)
:canonical: src.policies.look_ahead_aux.sans_opt.move_n_route_random

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.move_n_route_random
```
````

````{py:function} swap_n_route_random(routes_list, n=2)
:canonical: src.policies.look_ahead_aux.sans_opt.swap_n_route_random

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.swap_n_route_random
```
````

````{py:function} remove_n_bins_random(routes_list, removed_bins, bins_cannot_removed, n=2)
:canonical: src.policies.look_ahead_aux.sans_opt.remove_n_bins_random

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.remove_n_bins_random
```
````

````{py:function} add_n_bins_random(routes_list, removed_bins, stocks, vehicle_capacity, id_to_index, distance_matrix, n=2)
:canonical: src.policies.look_ahead_aux.sans_opt.add_n_bins_random

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.add_n_bins_random
```
````

````{py:function} add_route_with_removed_bins_random(routes_list, removed_bins, stocks, vehicle_capacity)
:canonical: src.policies.look_ahead_aux.sans_opt.add_route_with_removed_bins_random

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.add_route_with_removed_bins_random
```
````

````{py:function} move_n_route_consecutive(routes_list, n=2)
:canonical: src.policies.look_ahead_aux.sans_opt.move_n_route_consecutive

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.move_n_route_consecutive
```
````

````{py:function} swap_n_route_consecutive(routes_list, n=2)
:canonical: src.policies.look_ahead_aux.sans_opt.swap_n_route_consecutive

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.swap_n_route_consecutive
```
````

````{py:function} remove_n_bins_consecutive(routes_list, removed_bins, bins_cannot_removed, n=2)
:canonical: src.policies.look_ahead_aux.sans_opt.remove_n_bins_consecutive

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.remove_n_bins_consecutive
```
````

````{py:function} add_n_bins_consecutive(routes_list, removed_bins, stocks, vehicle_capacity, id_to_index, distance_matrix, n=2)
:canonical: src.policies.look_ahead_aux.sans_opt.add_n_bins_consecutive

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.add_n_bins_consecutive
```
````

````{py:function} add_route_with_removed_bins_consecutive(routes_list, removed_bins, stocks, vehicle_capacity)
:canonical: src.policies.look_ahead_aux.sans_opt.add_route_with_removed_bins_consecutive

```{autodoc2-docstring} src.policies.look_ahead_aux.sans_opt.add_route_with_removed_bins_consecutive
```
````
