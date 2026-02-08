# {py:mod}`src.policies.lin_kernighan_helsgaun`

```{py:module} src.policies.lin_kernighan_helsgaun
```

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_alpha_measures <src.policies.lin_kernighan_helsgaun.compute_alpha_measures>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun.compute_alpha_measures
    :summary:
    ```
* - {py:obj}`get_candidate_set <src.policies.lin_kernighan_helsgaun.get_candidate_set>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun.get_candidate_set
    :summary:
    ```
* - {py:obj}`calculate_penalty <src.policies.lin_kernighan_helsgaun.calculate_penalty>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun.calculate_penalty
    :summary:
    ```
* - {py:obj}`solve_lkh <src.policies.lin_kernighan_helsgaun.solve_lkh>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun.solve_lkh
    :summary:
    ```
````

### API

````{py:function} compute_alpha_measures(distance_matrix: numpy.ndarray) -> numpy.ndarray
:canonical: src.policies.lin_kernighan_helsgaun.compute_alpha_measures

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun.compute_alpha_measures
```
````

````{py:function} get_candidate_set(distance_matrix: numpy.ndarray, alpha_measures: numpy.ndarray, max_candidates: int = 5) -> typing.Dict[int, typing.List[int]]
:canonical: src.policies.lin_kernighan_helsgaun.get_candidate_set

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun.get_candidate_set
```
````

````{py:function} calculate_penalty(tour: typing.List[int], waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float]) -> float
:canonical: src.policies.lin_kernighan_helsgaun.calculate_penalty

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun.calculate_penalty
```
````

````{py:function} solve_lkh(distance_matrix: numpy.ndarray, initial_tour: typing.Optional[typing.List[int]] = None, max_iterations: int = 100, waste: typing.Optional[numpy.ndarray] = None, capacity: typing.Optional[float] = None) -> typing.Tuple[typing.List[int], float]
:canonical: src.policies.lin_kernighan_helsgaun.solve_lkh

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun.solve_lkh
```
````
