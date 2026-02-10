# {py:mod}`src.policies.simulated_annealing_neighborhood_search.select.consecutive`

```{py:module} src.policies.simulated_annealing_neighborhood_search.select.consecutive
```

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.select.consecutive
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_extract_valid_segment <src.policies.simulated_annealing_neighborhood_search.select.consecutive._extract_valid_segment>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.select.consecutive._extract_valid_segment
    :summary:
    ```
* - {py:obj}`remove_n_bins_consecutive <src.policies.simulated_annealing_neighborhood_search.select.consecutive.remove_n_bins_consecutive>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.select.consecutive.remove_n_bins_consecutive
    :summary:
    ```
* - {py:obj}`add_n_bins_consecutive <src.policies.simulated_annealing_neighborhood_search.select.consecutive.add_n_bins_consecutive>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.select.consecutive.add_n_bins_consecutive
    :summary:
    ```
* - {py:obj}`add_route_consecutive <src.policies.simulated_annealing_neighborhood_search.select.consecutive.add_route_consecutive>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.select.consecutive.add_route_consecutive
    :summary:
    ```
* - {py:obj}`add_route_with_removed_bins_consecutive <src.policies.simulated_annealing_neighborhood_search.select.consecutive.add_route_with_removed_bins_consecutive>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.select.consecutive.add_route_with_removed_bins_consecutive
    :summary:
    ```
````

### API

````{py:function} _extract_valid_segment(chosen_route, chosen_n, bins_cannot_removed)
:canonical: src.policies.simulated_annealing_neighborhood_search.select.consecutive._extract_valid_segment

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.select.consecutive._extract_valid_segment
```
````

````{py:function} remove_n_bins_consecutive(routes_list, removed_bins, bins_cannot_removed)
:canonical: src.policies.simulated_annealing_neighborhood_search.select.consecutive.remove_n_bins_consecutive

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.select.consecutive.remove_n_bins_consecutive
```
````

````{py:function} add_n_bins_consecutive(routes_list, removed_bins)
:canonical: src.policies.simulated_annealing_neighborhood_search.select.consecutive.add_n_bins_consecutive

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.select.consecutive.add_n_bins_consecutive
```
````

````{py:function} add_route_consecutive(routes_list, distance_matrix)
:canonical: src.policies.simulated_annealing_neighborhood_search.select.consecutive.add_route_consecutive

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.select.consecutive.add_route_consecutive
```
````

````{py:function} add_route_with_removed_bins_consecutive(routes_list, removed_bins, distance_matrix)
:canonical: src.policies.simulated_annealing_neighborhood_search.select.consecutive.add_route_with_removed_bins_consecutive

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.select.consecutive.add_route_with_removed_bins_consecutive
```
````
