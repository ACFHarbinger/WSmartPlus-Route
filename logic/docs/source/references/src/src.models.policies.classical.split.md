# {py:mod}`src.models.policies.classical.split`

```{py:module} src.models.policies.classical.split
```

```{autodoc2-docstring} src.models.policies.classical.split
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_linear_split <src.models.policies.classical.split.vectorized_linear_split>`
  - ```{autodoc2-docstring} src.models.policies.classical.split.vectorized_linear_split
    :summary:
    ```
* - {py:obj}`_vectorized_split_limited <src.models.policies.classical.split._vectorized_split_limited>`
  - ```{autodoc2-docstring} src.models.policies.classical.split._vectorized_split_limited
    :summary:
    ```
* - {py:obj}`_reconstruct_routes <src.models.policies.classical.split._reconstruct_routes>`
  - ```{autodoc2-docstring} src.models.policies.classical.split._reconstruct_routes
    :summary:
    ```
* - {py:obj}`_reconstruct_limited <src.models.policies.classical.split._reconstruct_limited>`
  - ```{autodoc2-docstring} src.models.policies.classical.split._reconstruct_limited
    :summary:
    ```
````

### API

````{py:function} vectorized_linear_split(giant_tours, dist_matrix, demands, vehicle_capacity, max_len=None, max_vehicles=None)
:canonical: src.models.policies.classical.split.vectorized_linear_split

```{autodoc2-docstring} src.models.policies.classical.split.vectorized_linear_split
```
````

````{py:function} _vectorized_split_limited(B, N, device, max_vehicles, capacity, cum_load, cum_dist_pad, d_0_i, d_i_0, giant_tours)
:canonical: src.models.policies.classical.split._vectorized_split_limited

```{autodoc2-docstring} src.models.policies.classical.split._vectorized_split_limited
```
````

````{py:function} _reconstruct_routes(B, N, giant_tours, P, costs)
:canonical: src.models.policies.classical.split._reconstruct_routes

```{autodoc2-docstring} src.models.policies.classical.split._reconstruct_routes
```
````

````{py:function} _reconstruct_limited(B, N, giant_tours, P_k, best_k, costs)
:canonical: src.models.policies.classical.split._reconstruct_limited

```{autodoc2-docstring} src.models.policies.classical.split._reconstruct_limited
```
````
