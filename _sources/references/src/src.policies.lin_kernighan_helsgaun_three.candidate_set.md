# {py:mod}`src.policies.lin_kernighan_helsgaun_three.candidate_set`

```{py:module} src.policies.lin_kernighan_helsgaun_three.candidate_set
```

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.candidate_set
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_compute_all_pairs_max_edge <src.policies.lin_kernighan_helsgaun_three.candidate_set._compute_all_pairs_max_edge>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.candidate_set._compute_all_pairs_max_edge
    :summary:
    ```
* - {py:obj}`compute_alpha_measures <src.policies.lin_kernighan_helsgaun_three.candidate_set.compute_alpha_measures>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.candidate_set.compute_alpha_measures
    :summary:
    ```
* - {py:obj}`get_candidate_set <src.policies.lin_kernighan_helsgaun_three.candidate_set.get_candidate_set>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.candidate_set.get_candidate_set
    :summary:
    ```
````

### API

````{py:function} _compute_all_pairs_max_edge(mst_adj: numpy.ndarray, n: int) -> numpy.ndarray
:canonical: src.policies.lin_kernighan_helsgaun_three.candidate_set._compute_all_pairs_max_edge

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.candidate_set._compute_all_pairs_max_edge
```
````

````{py:function} compute_alpha_measures(distance_matrix: numpy.ndarray, pi: typing.Optional[numpy.ndarray] = None) -> numpy.ndarray
:canonical: src.policies.lin_kernighan_helsgaun_three.candidate_set.compute_alpha_measures

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.candidate_set.compute_alpha_measures
```
````

````{py:function} get_candidate_set(distance_matrix: numpy.ndarray, alpha_measures: numpy.ndarray, max_candidates: int = 5) -> typing.Dict[int, typing.List[int]]
:canonical: src.policies.lin_kernighan_helsgaun_three.candidate_set.get_candidate_set

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.candidate_set.get_candidate_set
```
````
