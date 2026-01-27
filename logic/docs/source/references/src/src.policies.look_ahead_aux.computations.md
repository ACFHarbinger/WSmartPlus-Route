# {py:mod}`src.policies.look_ahead_aux.computations`

```{py:module} src.policies.look_ahead_aux.computations
```

```{autodoc2-docstring} src.policies.look_ahead_aux.computations
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_waste_collection_revenue <src.policies.look_ahead_aux.computations.compute_waste_collection_revenue>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_waste_collection_revenue
    :summary:
    ```
* - {py:obj}`compute_distance_per_route <src.policies.look_ahead_aux.computations.compute_distance_per_route>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_distance_per_route
    :summary:
    ```
* - {py:obj}`compute_route_time <src.policies.look_ahead_aux.computations.compute_route_time>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_route_time
    :summary:
    ```
* - {py:obj}`compute_transportation_cost <src.policies.look_ahead_aux.computations.compute_transportation_cost>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_transportation_cost
    :summary:
    ```
* - {py:obj}`compute_vehicle_use_penalty <src.policies.look_ahead_aux.computations.compute_vehicle_use_penalty>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_vehicle_use_penalty
    :summary:
    ```
* - {py:obj}`compute_route_time_difference_penalty <src.policies.look_ahead_aux.computations.compute_route_time_difference_penalty>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_route_time_difference_penalty
    :summary:
    ```
* - {py:obj}`compute_shift_excess_penalty <src.policies.look_ahead_aux.computations.compute_shift_excess_penalty>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_shift_excess_penalty
    :summary:
    ```
* - {py:obj}`compute_load_excess_penalty <src.policies.look_ahead_aux.computations.compute_load_excess_penalty>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_load_excess_penalty
    :summary:
    ```
* - {py:obj}`compute_profit <src.policies.look_ahead_aux.computations.compute_profit>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_profit
    :summary:
    ```
* - {py:obj}`compute_real_profit <src.policies.look_ahead_aux.computations.compute_real_profit>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_real_profit
    :summary:
    ```
* - {py:obj}`compute_total_profit <src.policies.look_ahead_aux.computations.compute_total_profit>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_total_profit
    :summary:
    ```
* - {py:obj}`compute_sans_route_cost <src.policies.look_ahead_aux.computations.compute_sans_route_cost>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_sans_route_cost
    :summary:
    ```
* - {py:obj}`compute_total_cost <src.policies.look_ahead_aux.computations.compute_total_cost>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_total_cost
    :summary:
    ```
````

### API

````{py:function} compute_waste_collection_revenue(routes_list, data, E, B, R)
:canonical: src.policies.look_ahead_aux.computations.compute_waste_collection_revenue

```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_waste_collection_revenue
```
````

````{py:function} compute_distance_per_route(routes_list, distance_matrix)
:canonical: src.policies.look_ahead_aux.computations.compute_distance_per_route

```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_distance_per_route
```
````

````{py:function} compute_route_time(routes_list, distance_route_vector)
:canonical: src.policies.look_ahead_aux.computations.compute_route_time

```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_route_time
```
````

````{py:function} compute_transportation_cost(routes_list, distance_route_vector, C)
:canonical: src.policies.look_ahead_aux.computations.compute_transportation_cost

```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_transportation_cost
```
````

````{py:function} compute_vehicle_use_penalty(routes_list, p_vehicle)
:canonical: src.policies.look_ahead_aux.computations.compute_vehicle_use_penalty

```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_vehicle_use_penalty
```
````

````{py:function} compute_route_time_difference_penalty(routes_list, p_route_difference, distance_route_vector)
:canonical: src.policies.look_ahead_aux.computations.compute_route_time_difference_penalty

```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_route_time_difference_penalty
```
````

````{py:function} compute_shift_excess_penalty(routes_list, p_shift, distance_matrix, distance_route_vector, shift_duration)
:canonical: src.policies.look_ahead_aux.computations.compute_shift_excess_penalty

```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_shift_excess_penalty
```
````

````{py:function} compute_load_excess_penalty(routes_list, p_load, data, vehicle_capacity)
:canonical: src.policies.look_ahead_aux.computations.compute_load_excess_penalty

```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_load_excess_penalty
```
````

````{py:function} compute_profit(routes_list, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)
:canonical: src.policies.look_ahead_aux.computations.compute_profit

```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_profit
```
````

````{py:function} compute_real_profit(routes_list, p_vehicle, data, distance_matrix, values)
:canonical: src.policies.look_ahead_aux.computations.compute_real_profit

```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_real_profit
```
````

````{py:function} compute_total_profit(routes, distance_matrix, id_to_index, data, R, V, density, cost_per_km=1.0)
:canonical: src.policies.look_ahead_aux.computations.compute_total_profit

```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_total_profit
```
````

````{py:function} compute_sans_route_cost(route, distance_matrix, id_to_index)
:canonical: src.policies.look_ahead_aux.computations.compute_sans_route_cost

```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_sans_route_cost
```
````

````{py:function} compute_total_cost(routes, distance_matrix, id_to_index)
:canonical: src.policies.look_ahead_aux.computations.compute_total_cost

```{autodoc2-docstring} src.policies.look_ahead_aux.computations.compute_total_cost
```
````
