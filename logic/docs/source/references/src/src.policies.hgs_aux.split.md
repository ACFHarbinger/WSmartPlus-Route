# {py:mod}`src.policies.hgs_aux.split`

```{py:module} src.policies.hgs_aux.split
```

```{autodoc2-docstring} src.policies.hgs_aux.split
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LinearSplit <src.policies.hgs_aux.split.LinearSplit>`
  - ```{autodoc2-docstring} src.policies.hgs_aux.split.LinearSplit
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`split_algorithm <src.policies.hgs_aux.split.split_algorithm>`
  - ```{autodoc2-docstring} src.policies.hgs_aux.split.split_algorithm
    :summary:
    ```
````

### API

`````{py:class} LinearSplit(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, R: float, C: float, max_vehicles: int = 0)
:canonical: src.policies.hgs_aux.split.LinearSplit

```{autodoc2-docstring} src.policies.hgs_aux.split.LinearSplit
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hgs_aux.split.LinearSplit.__init__
```

````{py:method} split(giant_tour: typing.List[int]) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.hgs_aux.split.LinearSplit.split

```{autodoc2-docstring} src.policies.hgs_aux.split.LinearSplit.split
```

````

````{py:method} _fallback_split(giant_tour: typing.List[int]) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.hgs_aux.split.LinearSplit._fallback_split

```{autodoc2-docstring} src.policies.hgs_aux.split.LinearSplit._fallback_split
```

````

````{py:method} _split_unlimited(n: int, nodes: typing.List[int], cum_load: typing.List[float], cum_rev: typing.List[float], cum_dist: typing.List[float], d_0_x: typing.List[float], d_x_0: typing.List[float]) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.hgs_aux.split.LinearSplit._split_unlimited

```{autodoc2-docstring} src.policies.hgs_aux.split.LinearSplit._split_unlimited
```

````

````{py:method} _split_limited(n: int, nodes: typing.List[int], cum_load: typing.List[float], cum_rev: typing.List[float], cum_dist: typing.List[float], d_0_x: typing.List[float], d_x_0: typing.List[float]) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.hgs_aux.split.LinearSplit._split_limited

```{autodoc2-docstring} src.policies.hgs_aux.split.LinearSplit._split_limited
```

````

````{py:method} _reconstruct_limited(n: int, nodes: typing.List[int], P: typing.List[typing.List[int]], k_opt: int, total_profit: float) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.hgs_aux.split.LinearSplit._reconstruct_limited

```{autodoc2-docstring} src.policies.hgs_aux.split.LinearSplit._reconstruct_limited
```

````

````{py:method} _reconstruct(n: int, nodes: typing.List[int], P: typing.List[int], total_profit: float) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.hgs_aux.split.LinearSplit._reconstruct

```{autodoc2-docstring} src.policies.hgs_aux.split.LinearSplit._reconstruct
```

````

`````

````{py:function} split_algorithm(giant_tour: typing.List[int], dist_matrix, demands, capacity, R, C, values)
:canonical: src.policies.hgs_aux.split.split_algorithm

```{autodoc2-docstring} src.policies.hgs_aux.split.split_algorithm
```
````
