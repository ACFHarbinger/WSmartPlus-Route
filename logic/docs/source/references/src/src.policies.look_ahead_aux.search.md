# {py:mod}`src.policies.look_ahead_aux.search`

```{py:module} src.policies.look_ahead_aux.search
```

```{autodoc2-docstring} src.policies.look_ahead_aux.search
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`local_search <src.policies.look_ahead_aux.search.local_search>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.search.local_search
    :summary:
    ```
* - {py:obj}`local_search_2 <src.policies.look_ahead_aux.search.local_search_2>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.search.local_search_2
    :summary:
    ```
* - {py:obj}`local_search_reversed <src.policies.look_ahead_aux.search.local_search_reversed>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.search.local_search_reversed
    :summary:
    ```
````

### API

````{py:function} local_search(routes_list, removed_bins, distance_matrix, bins_cannot_removed)
:canonical: src.policies.look_ahead_aux.search.local_search

```{autodoc2-docstring} src.policies.look_ahead_aux.search.local_search
```
````

````{py:function} local_search_2(previous_solution, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)
:canonical: src.policies.look_ahead_aux.search.local_search_2

```{autodoc2-docstring} src.policies.look_ahead_aux.search.local_search_2
```
````

````{py:function} local_search_reversed(previous_solution, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)
:canonical: src.policies.look_ahead_aux.search.local_search_reversed

```{autodoc2-docstring} src.policies.look_ahead_aux.search.local_search_reversed
```
````
