# {py:mod}`src.policies.simulated_annealing_neighborhood_search.heuristics.sans`

```{py:module} src.policies.simulated_annealing_neighborhood_search.heuristics.sans
```

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_initialize_solution_state <src.policies.simulated_annealing_neighborhood_search.heuristics.sans._initialize_solution_state>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans._initialize_solution_state
    :summary:
    ```
* - {py:obj}`_select_neighbor <src.policies.simulated_annealing_neighborhood_search.heuristics.sans._select_neighbor>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans._select_neighbor
    :summary:
    ```
* - {py:obj}`improved_simulated_annealing <src.policies.simulated_annealing_neighborhood_search.heuristics.sans.improved_simulated_annealing>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans.improved_simulated_annealing
    :summary:
    ```
````

### API

````{py:function} _initialize_solution_state(routes, id_to_index, distance_matrix, data)
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans._initialize_solution_state

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans._initialize_solution_state
```
````

````{py:function} _select_neighbor(solution, removed_bins, data, vehicle_capacity, id_to_index, stocks, must_go_bins, distance_matrix)
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans._select_neighbor

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans._select_neighbor
```
````

````{py:function} improved_simulated_annealing(routes, distance_matrix, time_limit, id_to_index, data, vehicle_capacity, T_init=1000, T_min=0.001, alpha=0.995, iterations_per_T=100, R=0.165, V=2.5, density=20, C=1.0, must_go_bins=None, removed_bins=None, verbose=False, perc_bins_can_overflow=0.0, volume=2.5, density_val=20, max_vehicles=None)
:canonical: src.policies.simulated_annealing_neighborhood_search.heuristics.sans.improved_simulated_annealing

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.heuristics.sans.improved_simulated_annealing
```
````
