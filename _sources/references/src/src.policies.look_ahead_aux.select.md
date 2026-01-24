# {py:mod}`src.policies.look_ahead_aux.select`

```{py:module} src.policies.look_ahead_aux.select
```

```{autodoc2-docstring} src.policies.look_ahead_aux.select
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`remove_bin <src.policies.look_ahead_aux.select.remove_bin>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.select.remove_bin
    :summary:
    ```
* - {py:obj}`add_bin <src.policies.look_ahead_aux.select.add_bin>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.select.add_bin
    :summary:
    ```
* - {py:obj}`insert_bins <src.policies.look_ahead_aux.select.insert_bins>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.select.insert_bins
    :summary:
    ```
* - {py:obj}`remove_bins_end <src.policies.look_ahead_aux.select.remove_bins_end>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.select.remove_bins_end
    :summary:
    ```
* - {py:obj}`remove_n_bins_random <src.policies.look_ahead_aux.select.remove_n_bins_random>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.select.remove_n_bins_random
    :summary:
    ```
* - {py:obj}`remove_n_bins_consecutive <src.policies.look_ahead_aux.select.remove_n_bins_consecutive>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.select.remove_n_bins_consecutive
    :summary:
    ```
* - {py:obj}`add_n_bins_random <src.policies.look_ahead_aux.select.add_n_bins_random>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.select.add_n_bins_random
    :summary:
    ```
* - {py:obj}`add_n_bins_consecutive <src.policies.look_ahead_aux.select.add_n_bins_consecutive>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.select.add_n_bins_consecutive
    :summary:
    ```
* - {py:obj}`add_route_random <src.policies.look_ahead_aux.select.add_route_random>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.select.add_route_random
    :summary:
    ```
* - {py:obj}`add_route_consecutive <src.policies.look_ahead_aux.select.add_route_consecutive>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.select.add_route_consecutive
    :summary:
    ```
* - {py:obj}`add_route_with_removed_bins_random <src.policies.look_ahead_aux.select.add_route_with_removed_bins_random>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.select.add_route_with_removed_bins_random
    :summary:
    ```
* - {py:obj}`add_route_with_removed_bins_consecutive <src.policies.look_ahead_aux.select.add_route_with_removed_bins_consecutive>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.select.add_route_with_removed_bins_consecutive
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <src.policies.look_ahead_aux.select.__all__>`
  - ```{autodoc2-docstring} src.policies.look_ahead_aux.select.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: src.policies.look_ahead_aux.select.__all__
:value: >
   ['remove_bin', 'add_bin', 'insert_bins', 'remove_bins_end', 'remove_n_bins_random', 'remove_n_bins_c...

```{autodoc2-docstring} src.policies.look_ahead_aux.select.__all__
```

````

````{py:function} remove_bin(routes_list, removed_bins, bins_cannot_removed)
:canonical: src.policies.look_ahead_aux.select.remove_bin

```{autodoc2-docstring} src.policies.look_ahead_aux.select.remove_bin
```
````

````{py:function} add_bin(routes_list, removed_bins)
:canonical: src.policies.look_ahead_aux.select.add_bin

```{autodoc2-docstring} src.policies.look_ahead_aux.select.add_bin
```
````

````{py:function} insert_bins(routes_list, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)
:canonical: src.policies.look_ahead_aux.select.insert_bins

```{autodoc2-docstring} src.policies.look_ahead_aux.select.insert_bins
```
````

````{py:function} remove_bins_end(routes_list, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data, bins_cannot_removed, distance_matrix, values)
:canonical: src.policies.look_ahead_aux.select.remove_bins_end

```{autodoc2-docstring} src.policies.look_ahead_aux.select.remove_bins_end
```
````

````{py:function} remove_n_bins_random(routes_list, removed_bins, bins_cannot_removed)
:canonical: src.policies.look_ahead_aux.select.remove_n_bins_random

```{autodoc2-docstring} src.policies.look_ahead_aux.select.remove_n_bins_random
```
````

````{py:function} remove_n_bins_consecutive(routes_list, removed_bins, bins_cannot_removed)
:canonical: src.policies.look_ahead_aux.select.remove_n_bins_consecutive

```{autodoc2-docstring} src.policies.look_ahead_aux.select.remove_n_bins_consecutive
```
````

````{py:function} add_n_bins_random(routes_list, removed_bins)
:canonical: src.policies.look_ahead_aux.select.add_n_bins_random

```{autodoc2-docstring} src.policies.look_ahead_aux.select.add_n_bins_random
```
````

````{py:function} add_n_bins_consecutive(routes_list, removed_bins)
:canonical: src.policies.look_ahead_aux.select.add_n_bins_consecutive

```{autodoc2-docstring} src.policies.look_ahead_aux.select.add_n_bins_consecutive
```
````

````{py:function} add_route_random(routes_list, distance_matrix)
:canonical: src.policies.look_ahead_aux.select.add_route_random

```{autodoc2-docstring} src.policies.look_ahead_aux.select.add_route_random
```
````

````{py:function} add_route_consecutive(routes_list, distance_matrix)
:canonical: src.policies.look_ahead_aux.select.add_route_consecutive

```{autodoc2-docstring} src.policies.look_ahead_aux.select.add_route_consecutive
```
````

````{py:function} add_route_with_removed_bins_random(routes_list, removed_bins, distance_matrix)
:canonical: src.policies.look_ahead_aux.select.add_route_with_removed_bins_random

```{autodoc2-docstring} src.policies.look_ahead_aux.select.add_route_with_removed_bins_random
```
````

````{py:function} add_route_with_removed_bins_consecutive(routes_list, removed_bins, distance_matrix)
:canonical: src.policies.look_ahead_aux.select.add_route_with_removed_bins_consecutive

```{autodoc2-docstring} src.policies.look_ahead_aux.select.add_route_with_removed_bins_consecutive
```
````
