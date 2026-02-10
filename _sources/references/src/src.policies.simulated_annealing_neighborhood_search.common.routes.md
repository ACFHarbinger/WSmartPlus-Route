# {py:mod}`src.policies.simulated_annealing_neighborhood_search.common.routes`

```{py:module} src.policies.simulated_annealing_neighborhood_search.common.routes
```

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.routes
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_points <src.policies.simulated_annealing_neighborhood_search.common.routes.create_points>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.routes.create_points
    :summary:
    ```
* - {py:obj}`_find_crossed_arcs <src.policies.simulated_annealing_neighborhood_search.common.routes._find_crossed_arcs>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.routes._find_crossed_arcs
    :summary:
    ```
* - {py:obj}`_remove_invalid_crossings <src.policies.simulated_annealing_neighborhood_search.common.routes._remove_invalid_crossings>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.routes._remove_invalid_crossings
    :summary:
    ```
* - {py:obj}`uncross_arcs_in_routes <src.policies.simulated_annealing_neighborhood_search.common.routes.uncross_arcs_in_routes>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.routes.uncross_arcs_in_routes
    :summary:
    ```
* - {py:obj}`rearrange_part_route <src.policies.simulated_annealing_neighborhood_search.common.routes.rearrange_part_route>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.routes.rearrange_part_route
    :summary:
    ```
* - {py:obj}`organize_route <src.policies.simulated_annealing_neighborhood_search.common.routes.organize_route>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.routes.organize_route
    :summary:
    ```
* - {py:obj}`two_opt_uncross_arc <src.policies.simulated_annealing_neighborhood_search.common.routes.two_opt_uncross_arc>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.routes.two_opt_uncross_arc
    :summary:
    ```
* - {py:obj}`uncross_arcs_in_sans_routes <src.policies.simulated_annealing_neighborhood_search.common.routes.uncross_arcs_in_sans_routes>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.routes.uncross_arcs_in_sans_routes
    :summary:
    ```
````

### API

````{py:function} create_points(data, bins_coordinates)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.routes.create_points

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.routes.create_points
```
````

````{py:function} _find_crossed_arcs(route, points, route_idx, cache)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.routes._find_crossed_arcs

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.routes._find_crossed_arcs
```
````

````{py:function} _remove_invalid_crossings(crossings, points)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.routes._remove_invalid_crossings

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.routes._remove_invalid_crossings
```
````

````{py:function} uncross_arcs_in_routes(previous_solution, p_vehicle, p_load, p_route_difference, p_shift, data, points, distance_matrix, values)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.routes.uncross_arcs_in_routes

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.routes.uncross_arcs_in_routes
```
````

````{py:function} rearrange_part_route(routes_list, distance_matrix)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.routes.rearrange_part_route

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.routes.rearrange_part_route
```
````

````{py:function} organize_route(bins, distance_matrix)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.routes.organize_route

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.routes.organize_route
```
````

````{py:function} two_opt_uncross_arc(route, distance_matrix, id_to_index)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.routes.two_opt_uncross_arc

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.routes.two_opt_uncross_arc
```
````

````{py:function} uncross_arcs_in_sans_routes(routes, id_to_index, distance_matrix)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.routes.uncross_arcs_in_sans_routes

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.routes.uncross_arcs_in_sans_routes
```
````
