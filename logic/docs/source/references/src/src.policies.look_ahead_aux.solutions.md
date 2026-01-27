# {py:mod}`src.policies.look_ahead_aux.solutions`

```{py:module} src.policies.look_ahead_aux.solutions
```

```{autodoc2-docstring} src.policies.look_ahead_aux.solutions
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`find_initial_solution <src.policies.look_ahead_aux.solutions.find_initial_solution>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.solutions.find_initial_solution
    :summary:
    ```
* - {py:obj}`find_solutions <src.policies.look_ahead_aux.solutions.find_solutions>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.solutions.find_solutions
    :summary:
    ```
* - {py:obj}`compute_initial_solution <src.policies.look_ahead_aux.solutions.compute_initial_solution>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.solutions.compute_initial_solution
    :summary:
    ```
* - {py:obj}`improved_simulated_annealing <src.policies.look_ahead_aux.solutions.improved_simulated_annealing>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.solutions.improved_simulated_annealing
    :summary:
    ```
````

### API

````{py:function} find_initial_solution(data, bins_coordinates, distance_matrix, number_of_bins, vehicle_capacity, E, B)
:canonical: src.policies.look_ahead_aux.solutions.find_initial_solution

```{autodoc2-docstring} src.policies.look_ahead_aux.solutions.find_initial_solution
```
````

````{py:function} find_solutions(data, bins_coordinates, distance_matrix, chosen_combination, must_go_bins, values, n_bins, points, time_limit)
:canonical: src.policies.look_ahead_aux.solutions.find_solutions

```{autodoc2-docstring} src.policies.look_ahead_aux.solutions.find_solutions
```
````

````{py:function} compute_initial_solution(data, bins_coordinates, distance_matrix, vehicle_capacity, id_to_index)
:canonical: src.policies.look_ahead_aux.solutions.compute_initial_solution

```{autodoc2-docstring} src.policies.look_ahead_aux.solutions.compute_initial_solution
```
````

````{py:function} improved_simulated_annealing(routes, distance_matrix, time_limit, id_to_index, data, vehicle_capacity, T_init=1000, T_min=0.001, alpha=0.995, iterations_per_T=100, R=0.165, V=2.5, density=20, C=1.0, must_go_bins=None, removed_bins=None, verbose=False, perc_bins_can_overflow=0.0, volume=2.5, density_val=20, max_vehicles=None)
:canonical: src.policies.look_ahead_aux.solutions.improved_simulated_annealing

```{autodoc2-docstring} src.policies.look_ahead_aux.solutions.improved_simulated_annealing
```
````
