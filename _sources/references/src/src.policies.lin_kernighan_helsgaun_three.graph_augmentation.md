# {py:mod}`src.policies.lin_kernighan_helsgaun_three.graph_augmentation`

```{py:module} src.policies.lin_kernighan_helsgaun_three.graph_augmentation
```

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.graph_augmentation
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`augment_graph <src.policies.lin_kernighan_helsgaun_three.graph_augmentation.augment_graph>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.graph_augmentation.augment_graph
    :summary:
    ```
* - {py:obj}`decode_augmented_tour <src.policies.lin_kernighan_helsgaun_three.graph_augmentation.decode_augmented_tour>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.graph_augmentation.decode_augmented_tour
    :summary:
    ```
* - {py:obj}`is_dummy_depot <src.policies.lin_kernighan_helsgaun_three.graph_augmentation.is_dummy_depot>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.graph_augmentation.is_dummy_depot
    :summary:
    ```
* - {py:obj}`is_any_depot <src.policies.lin_kernighan_helsgaun_three.graph_augmentation.is_any_depot>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.graph_augmentation.is_any_depot
    :summary:
    ```
* - {py:obj}`inject_augmented_dummies <src.policies.lin_kernighan_helsgaun_three.graph_augmentation.inject_augmented_dummies>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.graph_augmentation.inject_augmented_dummies
    :summary:
    ```
* - {py:obj}`validate_augmented_graph <src.policies.lin_kernighan_helsgaun_three.graph_augmentation.validate_augmented_graph>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.graph_augmentation.validate_augmented_graph
    :summary:
    ```
````

### API

````{py:function} augment_graph(distance_matrix: numpy.ndarray, wastes: typing.Dict[int, float], n_vehicles: int, capacity: float = 100.0, high_penalty: float = DEFAULT_HIGH_PENALTY) -> typing.Tuple[numpy.ndarray, numpy.ndarray, int]
:canonical: src.policies.lin_kernighan_helsgaun_three.graph_augmentation.augment_graph

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.graph_augmentation.augment_graph
```
````

````{py:function} decode_augmented_tour(tour: typing.List[int], n_original: int) -> typing.List[typing.List[int]]
:canonical: src.policies.lin_kernighan_helsgaun_three.graph_augmentation.decode_augmented_tour

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.graph_augmentation.decode_augmented_tour
```
````

````{py:function} is_dummy_depot(node: int, n_original: int) -> bool
:canonical: src.policies.lin_kernighan_helsgaun_three.graph_augmentation.is_dummy_depot

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.graph_augmentation.is_dummy_depot
```
````

````{py:function} is_any_depot(node: int, n_original: int) -> bool
:canonical: src.policies.lin_kernighan_helsgaun_three.graph_augmentation.is_any_depot

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.graph_augmentation.is_any_depot
```
````

````{py:function} inject_augmented_dummies(routes: typing.List[typing.List[int]], n_original: int, n_vehicles: int) -> typing.List[int]
:canonical: src.policies.lin_kernighan_helsgaun_three.graph_augmentation.inject_augmented_dummies

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.graph_augmentation.inject_augmented_dummies
```
````

````{py:function} validate_augmented_graph(augmented_dist: numpy.ndarray, augmented_waste: numpy.ndarray, n_original: int, n_vehicles: int) -> None
:canonical: src.policies.lin_kernighan_helsgaun_three.graph_augmentation.validate_augmented_graph

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.graph_augmentation.validate_augmented_graph
```
````
