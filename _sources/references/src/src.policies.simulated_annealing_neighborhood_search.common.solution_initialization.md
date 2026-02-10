# {py:mod}`src.policies.simulated_annealing_neighborhood_search.common.solution_initialization`

```{py:module} src.policies.simulated_annealing_neighborhood_search.common.solution_initialization
```

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.solution_initialization
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_categorize_bins_by_zone <src.policies.simulated_annealing_neighborhood_search.common.solution_initialization._categorize_bins_by_zone>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.solution_initialization._categorize_bins_by_zone
    :summary:
    ```
* - {py:obj}`_get_bin_stock <src.policies.simulated_annealing_neighborhood_search.common.solution_initialization._get_bin_stock>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.solution_initialization._get_bin_stock
    :summary:
    ```
* - {py:obj}`_find_closest_valid_bin <src.policies.simulated_annealing_neighborhood_search.common.solution_initialization._find_closest_valid_bin>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.solution_initialization._find_closest_valid_bin
    :summary:
    ```
* - {py:obj}`find_initial_solution <src.policies.simulated_annealing_neighborhood_search.common.solution_initialization.find_initial_solution>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.solution_initialization.find_initial_solution
    :summary:
    ```
* - {py:obj}`compute_initial_solution <src.policies.simulated_annealing_neighborhood_search.common.solution_initialization.compute_initial_solution>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.solution_initialization.compute_initial_solution
    :summary:
    ```
````

### API

````{py:function} _categorize_bins_by_zone(bins, bins_coordinates)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.solution_initialization._categorize_bins_by_zone

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.solution_initialization._categorize_bins_by_zone
```
````

````{py:function} _get_bin_stock(data, bin_id, E, B)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.solution_initialization._get_bin_stock

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.solution_initialization._get_bin_stock
```
````

````{py:function} _find_closest_valid_bin(current_bin, potential_bins_in_zone, available_bins, current_route, distance_matrix)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.solution_initialization._find_closest_valid_bin

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.solution_initialization._find_closest_valid_bin
```
````

````{py:function} find_initial_solution(data, bins_coordinates, distance_matrix, number_of_bins, vehicle_capacity, E, B)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.solution_initialization.find_initial_solution

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.solution_initialization.find_initial_solution
```
````

````{py:function} compute_initial_solution(data, bins_coordinates, distance_matrix, vehicle_capacity, id_to_index)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.solution_initialization.compute_initial_solution

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.solution_initialization.compute_initial_solution
```
````
