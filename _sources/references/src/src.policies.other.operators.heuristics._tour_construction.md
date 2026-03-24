# {py:mod}`src.policies.other.operators.heuristics._tour_construction`

```{py:module} src.policies.other.operators.heuristics._tour_construction
```

```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_construction
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`merge_tours <src.policies.other.operators.heuristics._tour_construction.merge_tours>`
  - ```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_construction.merge_tours
    :summary:
    ```
* - {py:obj}`_double_bridge_kick <src.policies.other.operators.heuristics._tour_construction._double_bridge_kick>`
  - ```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_construction._double_bridge_kick
    :summary:
    ```
* - {py:obj}`_initialize_tour <src.policies.other.operators.heuristics._tour_construction._initialize_tour>`
  - ```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_construction._initialize_tour
    :summary:
    ```
* - {py:obj}`_2opt_gain <src.policies.other.operators.heuristics._tour_construction._2opt_gain>`
  - ```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_construction._2opt_gain
    :summary:
    ```
* - {py:obj}`_3opt_gains <src.policies.other.operators.heuristics._tour_construction._3opt_gains>`
  - ```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_construction._3opt_gains
    :summary:
    ```
* - {py:obj}`_4opt_gains <src.policies.other.operators.heuristics._tour_construction._4opt_gains>`
  - ```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_construction._4opt_gains
    :summary:
    ```
* - {py:obj}`_5opt_gains <src.policies.other.operators.heuristics._tour_construction._5opt_gains>`
  - ```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_construction._5opt_gains
    :summary:
    ```
````

### API

````{py:function} merge_tours(tour1: typing.List[int], tour2: typing.List[int], distance_matrix: numpy.ndarray) -> typing.List[int]
:canonical: src.policies.other.operators.heuristics._tour_construction.merge_tours

```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_construction.merge_tours
```
````

````{py:function} _double_bridge_kick(tour: typing.List[int], distance_matrix: numpy.ndarray, rng: random.Random) -> typing.List[int]
:canonical: src.policies.other.operators.heuristics._tour_construction._double_bridge_kick

```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_construction._double_bridge_kick
```
````

````{py:function} _initialize_tour(distance_matrix: numpy.ndarray, initial_tour: typing.Optional[typing.List[int]]) -> typing.List[int]
:canonical: src.policies.other.operators.heuristics._tour_construction._initialize_tour

```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_construction._initialize_tour
```
````

````{py:function} _2opt_gain(t1: int, t2: int, t3: int, t4: int, d: numpy.ndarray) -> float
:canonical: src.policies.other.operators.heuristics._tour_construction._2opt_gain

```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_construction._2opt_gain
```
````

````{py:function} _3opt_gains(t1: int, t2: int, t3: int, t4: int, t5: int, t6: int, d: numpy.ndarray) -> typing.List[float]
:canonical: src.policies.other.operators.heuristics._tour_construction._3opt_gains

```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_construction._3opt_gains
```
````

````{py:function} _4opt_gains(t1: int, t2: int, t3: int, t4: int, t5: int, t6: int, t7: int, t8: int, d: numpy.ndarray) -> typing.List[float]
:canonical: src.policies.other.operators.heuristics._tour_construction._4opt_gains

```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_construction._4opt_gains
```
````

````{py:function} _5opt_gains(t1: int, t2: int, t3: int, t4: int, t5: int, t6: int, t7: int, t8: int, t9: int, t10: int, d: numpy.ndarray) -> typing.List[float]
:canonical: src.policies.other.operators.heuristics._tour_construction._5opt_gains

```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_construction._5opt_gains
```
````
