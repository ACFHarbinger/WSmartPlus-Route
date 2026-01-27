# {py:mod}`src.policies.look_ahead_aux.routes`

```{py:module} src.policies.look_ahead_aux.routes
```

```{autodoc2-docstring} src.policies.look_ahead_aux.routes
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_points <src.policies.look_ahead_aux.routes.create_points>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.routes.create_points
    :summary:
    ```
* - {py:obj}`uncross_arcs_in_routes <src.policies.look_ahead_aux.routes.uncross_arcs_in_routes>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.routes.uncross_arcs_in_routes
    :summary:
    ```
* - {py:obj}`rearrange_part_route <src.policies.look_ahead_aux.routes.rearrange_part_route>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.routes.rearrange_part_route
    :summary:
    ```
* - {py:obj}`organize_route <src.policies.look_ahead_aux.routes.organize_route>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.routes.organize_route
    :summary:
    ```
* - {py:obj}`two_opt_uncross_arc <src.policies.look_ahead_aux.routes.two_opt_uncross_arc>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.routes.two_opt_uncross_arc
    :summary:
    ```
* - {py:obj}`uncross_arcs_in_sans_routes <src.policies.look_ahead_aux.routes.uncross_arcs_in_sans_routes>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.routes.uncross_arcs_in_sans_routes
    :summary:
    ```
````

### API

````{py:function} create_points(data, bins_coordinates)
:canonical: src.policies.look_ahead_aux.routes.create_points

```{autodoc2-docstring} src.policies.look_ahead_aux.routes.create_points
```
````

````{py:function} uncross_arcs_in_routes(previous_solution, p_vehicle, p_load, p_route_difference, p_shift, data, points, distance_matrix, values)
:canonical: src.policies.look_ahead_aux.routes.uncross_arcs_in_routes

```{autodoc2-docstring} src.policies.look_ahead_aux.routes.uncross_arcs_in_routes
```
````

````{py:function} rearrange_part_route(routes_list, distance_matrix)
:canonical: src.policies.look_ahead_aux.routes.rearrange_part_route

```{autodoc2-docstring} src.policies.look_ahead_aux.routes.rearrange_part_route
```
````

````{py:function} organize_route(bins, distance_matrix)
:canonical: src.policies.look_ahead_aux.routes.organize_route

```{autodoc2-docstring} src.policies.look_ahead_aux.routes.organize_route
```
````

````{py:function} two_opt_uncross_arc(route, distance_matrix, id_to_index)
:canonical: src.policies.look_ahead_aux.routes.two_opt_uncross_arc

```{autodoc2-docstring} src.policies.look_ahead_aux.routes.two_opt_uncross_arc
```
````

````{py:function} uncross_arcs_in_sans_routes(routes, id_to_index, distance_matrix)
:canonical: src.policies.look_ahead_aux.routes.uncross_arcs_in_sans_routes

```{autodoc2-docstring} src.policies.look_ahead_aux.routes.uncross_arcs_in_sans_routes
```
````
