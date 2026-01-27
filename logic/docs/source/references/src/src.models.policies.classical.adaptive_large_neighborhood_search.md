# {py:mod}`src.models.policies.classical.adaptive_large_neighborhood_search`

```{py:module} src.models.policies.classical.adaptive_large_neighborhood_search
```

```{autodoc2-docstring} src.models.policies.classical.adaptive_large_neighborhood_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedALNS <src.models.policies.classical.adaptive_large_neighborhood_search.VectorizedALNS>`
  - ```{autodoc2-docstring} src.models.policies.classical.adaptive_large_neighborhood_search.VectorizedALNS
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`vectorized_random_removal <src.models.policies.classical.adaptive_large_neighborhood_search.vectorized_random_removal>`
  - ```{autodoc2-docstring} src.models.policies.classical.adaptive_large_neighborhood_search.vectorized_random_removal
    :summary:
    ```
* - {py:obj}`vectorized_worst_removal <src.models.policies.classical.adaptive_large_neighborhood_search.vectorized_worst_removal>`
  - ```{autodoc2-docstring} src.models.policies.classical.adaptive_large_neighborhood_search.vectorized_worst_removal
    :summary:
    ```
* - {py:obj}`vectorized_greedy_insertion <src.models.policies.classical.adaptive_large_neighborhood_search.vectorized_greedy_insertion>`
  - ```{autodoc2-docstring} src.models.policies.classical.adaptive_large_neighborhood_search.vectorized_greedy_insertion
    :summary:
    ```
````

### API

````{py:function} vectorized_random_removal(tours, n_remove)
:canonical: src.models.policies.classical.adaptive_large_neighborhood_search.vectorized_random_removal

```{autodoc2-docstring} src.models.policies.classical.adaptive_large_neighborhood_search.vectorized_random_removal
```
````

````{py:function} vectorized_worst_removal(tours, dist_matrix, n_remove, p=3)
:canonical: src.models.policies.classical.adaptive_large_neighborhood_search.vectorized_worst_removal

```{autodoc2-docstring} src.models.policies.classical.adaptive_large_neighborhood_search.vectorized_worst_removal
```
````

````{py:function} vectorized_greedy_insertion(partial_tours, removed_nodes, dist_matrix)
:canonical: src.models.policies.classical.adaptive_large_neighborhood_search.vectorized_greedy_insertion

```{autodoc2-docstring} src.models.policies.classical.adaptive_large_neighborhood_search.vectorized_greedy_insertion
```
````

`````{py:class} VectorizedALNS(dist_matrix, demands, vehicle_capacity, time_limit=1.0, device='cuda')
:canonical: src.models.policies.classical.adaptive_large_neighborhood_search.VectorizedALNS

```{autodoc2-docstring} src.models.policies.classical.adaptive_large_neighborhood_search.VectorizedALNS
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.policies.classical.adaptive_large_neighborhood_search.VectorizedALNS.__init__
```

````{py:method} solve(initial_solutions, n_iterations=100, time_limit=None)
:canonical: src.models.policies.classical.adaptive_large_neighborhood_search.VectorizedALNS.solve

```{autodoc2-docstring} src.models.policies.classical.adaptive_large_neighborhood_search.VectorizedALNS.solve
```

````

`````
