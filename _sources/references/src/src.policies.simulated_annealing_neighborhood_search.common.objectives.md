# {py:mod}`src.policies.simulated_annealing_neighborhood_search.common.objectives`

```{py:module} src.policies.simulated_annealing_neighborhood_search.common.objectives
```

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.objectives
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_profit <src.policies.simulated_annealing_neighborhood_search.common.objectives.compute_profit>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.objectives.compute_profit
    :summary:
    ```
* - {py:obj}`compute_real_profit <src.policies.simulated_annealing_neighborhood_search.common.objectives.compute_real_profit>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.objectives.compute_real_profit
    :summary:
    ```
* - {py:obj}`compute_total_profit <src.policies.simulated_annealing_neighborhood_search.common.objectives.compute_total_profit>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.objectives.compute_total_profit
    :summary:
    ```
````

### API

````{py:function} compute_profit(routes_list, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.objectives.compute_profit

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.objectives.compute_profit
```
````

````{py:function} compute_real_profit(routes_list, p_vehicle, data, distance_matrix, values)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.objectives.compute_real_profit

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.objectives.compute_real_profit
```
````

````{py:function} compute_total_profit(routes, distance_matrix, id_to_index, data, R, V, density, cost_per_km=1.0)
:canonical: src.policies.simulated_annealing_neighborhood_search.common.objectives.compute_total_profit

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.common.objectives.compute_total_profit
```
````
