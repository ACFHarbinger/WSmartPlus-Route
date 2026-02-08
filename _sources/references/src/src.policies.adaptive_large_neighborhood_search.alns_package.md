# {py:mod}`src.policies.adaptive_large_neighborhood_search.alns_package`

```{py:module} src.policies.adaptive_large_neighborhood_search.alns_package
```

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns_package
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ALNSState <src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState>`
  - ```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`alns_pkg_random_removal <src.policies.adaptive_large_neighborhood_search.alns_package.alns_pkg_random_removal>`
  - ```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns_package.alns_pkg_random_removal
    :summary:
    ```
* - {py:obj}`alns_pkg_worst_removal <src.policies.adaptive_large_neighborhood_search.alns_package.alns_pkg_worst_removal>`
  - ```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns_package.alns_pkg_worst_removal
    :summary:
    ```
* - {py:obj}`alns_pkg_greedy_insertion <src.policies.adaptive_large_neighborhood_search.alns_package.alns_pkg_greedy_insertion>`
  - ```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns_package.alns_pkg_greedy_insertion
    :summary:
    ```
* - {py:obj}`run_alns_package <src.policies.adaptive_large_neighborhood_search.alns_package.run_alns_package>`
  - ```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns_package.run_alns_package
    :summary:
    ```
````

### API

`````{py:class} ALNSState(routes: typing.List[typing.List[int]], unassigned: typing.List[int], dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, R: float, C: float, values: typing.Dict[str, typing.Any])
:canonical: src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState.__init__
```

````{py:method} copy() -> src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState
:canonical: src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState.copy

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState.copy
```

````

````{py:method} objective() -> float
:canonical: src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState.objective

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState.objective
```

````

````{py:method} calculate_profit() -> float
:canonical: src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState.calculate_profit

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState.calculate_profit
```

````

````{py:property} cost
:canonical: src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState.cost
:type: float

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState.cost
```

````

`````

````{py:function} alns_pkg_random_removal(state: src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState, random_state: numpy.random.RandomState) -> src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState
:canonical: src.policies.adaptive_large_neighborhood_search.alns_package.alns_pkg_random_removal

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns_package.alns_pkg_random_removal
```
````

````{py:function} alns_pkg_worst_removal(state: src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState, random_state: numpy.random.RandomState) -> src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState
:canonical: src.policies.adaptive_large_neighborhood_search.alns_package.alns_pkg_worst_removal

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns_package.alns_pkg_worst_removal
```
````

````{py:function} alns_pkg_greedy_insertion(state: src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState, random_state: numpy.random.RandomState) -> src.policies.adaptive_large_neighborhood_search.alns_package.ALNSState
:canonical: src.policies.adaptive_large_neighborhood_search.alns_package.alns_pkg_greedy_insertion

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns_package.alns_pkg_greedy_insertion
```
````

````{py:function} run_alns_package(dist_matrix, demands, capacity, R, C, values)
:canonical: src.policies.adaptive_large_neighborhood_search.alns_package.run_alns_package

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns_package.run_alns_package
```
````
