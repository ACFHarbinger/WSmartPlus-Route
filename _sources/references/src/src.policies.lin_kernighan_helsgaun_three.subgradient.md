# {py:mod}`src.policies.lin_kernighan_helsgaun_three.subgradient`

```{py:module} src.policies.lin_kernighan_helsgaun_three.subgradient
```

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.subgradient
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_min_1_tree <src.policies.lin_kernighan_helsgaun_three.subgradient.compute_min_1_tree>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.subgradient.compute_min_1_tree
    :summary:
    ```
* - {py:obj}`solve_subgradient <src.policies.lin_kernighan_helsgaun_three.subgradient.solve_subgradient>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.subgradient.solve_subgradient
    :summary:
    ```
````

### API

````{py:function} compute_min_1_tree(distance_matrix: numpy.ndarray, pi: numpy.ndarray) -> typing.Tuple[float, numpy.ndarray, numpy.ndarray]
:canonical: src.policies.lin_kernighan_helsgaun_three.subgradient.compute_min_1_tree

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.subgradient.compute_min_1_tree
```
````

````{py:function} solve_subgradient(distance_matrix: numpy.ndarray, max_iterations: int = 200, n_original: typing.Optional[int] = None, initial_pi: typing.Optional[numpy.ndarray] = None) -> numpy.ndarray
:canonical: src.policies.lin_kernighan_helsgaun_three.subgradient.solve_subgradient

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.subgradient.solve_subgradient
```
````
