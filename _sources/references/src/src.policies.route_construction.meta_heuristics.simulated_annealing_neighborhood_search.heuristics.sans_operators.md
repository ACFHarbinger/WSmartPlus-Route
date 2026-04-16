# {py:mod}`src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators`

```{py:module} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_handle_2opt <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_2opt>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_2opt
    :summary:
    ```
* - {py:obj}`_handle_move <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_move>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_move
    :summary:
    ```
* - {py:obj}`_handle_swap <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_swap>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_swap
    :summary:
    ```
* - {py:obj}`_handle_insert <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_insert>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_insert
    :summary:
    ```
* - {py:obj}`apply_operator <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators.apply_operator>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators.apply_operator
    :summary:
    ```
````

### API

````{py:function} _handle_2opt(solution, rng)
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_2opt

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_2opt
```
````

````{py:function} _handle_move(solution, data, vehicle_capacity, id_to_index, rng)
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_move

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_move
```
````

````{py:function} _handle_swap(solution, rng)
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_swap

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_swap
```
````

````{py:function} _handle_insert(solution, data, stocks, vehicle_capacity, id_to_index, distance_matrix, candidate_removed_bins, rng)
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_insert

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_insert
```
````

````{py:function} apply_operator(op, new_solution, candidate_removed_bins, data, vehicle_capacity, id_to_index, stocks, mandatory_bins, distance_matrix, rng)
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators.apply_operator

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators.apply_operator
```
````
