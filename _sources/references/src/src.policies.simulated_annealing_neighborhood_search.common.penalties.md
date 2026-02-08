# {py:mod}`src.policies.simulated_annealing_neighborhood_search.common.penalties`

```{py:module} src.policies.simulated_annealing_neighborhood_search.common.penalties
```

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.penalties
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_transportation_cost <src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_transportation_cost>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_transportation_cost
    :summary:
    ```
* - {py:obj}`compute_vehicle_use_penalty <src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_vehicle_use_penalty>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_vehicle_use_penalty
    :summary:
    ```
* - {py:obj}`compute_route_time_difference_penalty <src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_route_time_difference_penalty>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_route_time_difference_penalty
    :summary:
    ```
* - {py:obj}`compute_shift_excess_penalty <src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_shift_excess_penalty>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_shift_excess_penalty
    :summary:
    ```
* - {py:obj}`compute_load_excess_penalty <src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_load_excess_penalty>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_load_excess_penalty
    :summary:
    ```
````

### API

````{py:function} compute_transportation_cost(routes_list, distance_route_vector, C)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_transportation_cost

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_transportation_cost
```
````

````{py:function} compute_vehicle_use_penalty(routes_list, p_vehicle)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_vehicle_use_penalty

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_vehicle_use_penalty
```
````

````{py:function} compute_route_time_difference_penalty(routes_list, p_route_difference, distance_route_vector)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_route_time_difference_penalty

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_route_time_difference_penalty
```
````

````{py:function} compute_shift_excess_penalty(routes_list, p_shift, distance_matrix, distance_route_vector, shift_duration)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_shift_excess_penalty

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_shift_excess_penalty
```
````

````{py:function} compute_load_excess_penalty(routes_list, p_load, data, vehicle_capacity)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_load_excess_penalty

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.penalties.compute_load_excess_penalty
```
````
