# {py:mod}`src.policies.lin_kernighan_helsgaun_three.objective`

```{py:module} src.policies.lin_kernighan_helsgaun_three.objective
```

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.objective
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`is_dummy_depot <src.policies.lin_kernighan_helsgaun_three.objective.is_dummy_depot>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.objective.is_dummy_depot
    :summary:
    ```
* - {py:obj}`is_any_depot <src.policies.lin_kernighan_helsgaun_three.objective.is_any_depot>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.objective.is_any_depot
    :summary:
    ```
* - {py:obj}`split_tour_at_dummies <src.policies.lin_kernighan_helsgaun_three.objective.split_tour_at_dummies>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.objective.split_tour_at_dummies
    :summary:
    ```
* - {py:obj}`calculate_penalty <src.policies.lin_kernighan_helsgaun_three.objective.calculate_penalty>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.objective.calculate_penalty
    :summary:
    ```
* - {py:obj}`get_score <src.policies.lin_kernighan_helsgaun_three.objective.get_score>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.objective.get_score
    :summary:
    ```
* - {py:obj}`is_better <src.policies.lin_kernighan_helsgaun_three.objective.is_better>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.objective.is_better
    :summary:
    ```
* - {py:obj}`penalty_delta <src.policies.lin_kernighan_helsgaun_three.objective.penalty_delta>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.objective.penalty_delta
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DEPOT_NODE <src.policies.lin_kernighan_helsgaun_three.objective.DEPOT_NODE>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.objective.DEPOT_NODE
    :summary:
    ```
````

### API

````{py:data} DEPOT_NODE
:canonical: src.policies.lin_kernighan_helsgaun_three.objective.DEPOT_NODE
:value: >
   0

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.objective.DEPOT_NODE
```

````

````{py:function} is_dummy_depot(node: int, n_original: typing.Optional[int] = None) -> bool
:canonical: src.policies.lin_kernighan_helsgaun_three.objective.is_dummy_depot

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.objective.is_dummy_depot
```
````

````{py:function} is_any_depot(node: int, n_original: typing.Optional[int] = None) -> bool
:canonical: src.policies.lin_kernighan_helsgaun_three.objective.is_any_depot

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.objective.is_any_depot
```
````

````{py:function} split_tour_at_dummies(tour: typing.List[int], n_original: typing.Optional[int] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.lin_kernighan_helsgaun_three.objective.split_tour_at_dummies

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.objective.split_tour_at_dummies
```
````

````{py:function} calculate_penalty(tour: typing.List[int], waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float], n_original: typing.Optional[int] = None) -> float
:canonical: src.policies.lin_kernighan_helsgaun_three.objective.calculate_penalty

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.objective.calculate_penalty
```
````

````{py:function} get_score(tour: typing.List[int], distance_matrix: numpy.ndarray, waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float], n_original: typing.Optional[int] = None) -> typing.Tuple[float, float]
:canonical: src.policies.lin_kernighan_helsgaun_three.objective.get_score

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.objective.get_score
```
````

````{py:function} is_better(p1: float, c1: float, p2: float, c2: float) -> bool
:canonical: src.policies.lin_kernighan_helsgaun_three.objective.is_better

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.objective.is_better
```
````

````{py:function} penalty_delta(old_tour: typing.List[int], new_tour: typing.List[int], waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float], n_original: typing.Optional[int] = None) -> float
:canonical: src.policies.lin_kernighan_helsgaun_three.objective.penalty_delta

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.objective.penalty_delta
```
````
