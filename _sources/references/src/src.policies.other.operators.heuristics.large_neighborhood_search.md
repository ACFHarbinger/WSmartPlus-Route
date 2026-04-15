# {py:mod}`src.policies.other.operators.heuristics.large_neighborhood_search`

```{py:module} src.policies.other.operators.heuristics.large_neighborhood_search
```

```{autodoc2-docstring} src.policies.other.operators.heuristics.large_neighborhood_search
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`apply_lns <src.policies.other.operators.heuristics.large_neighborhood_search.apply_lns>`
  - ```{autodoc2-docstring} src.policies.other.operators.heuristics.large_neighborhood_search.apply_lns
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_DESTROY_OPS <src.policies.other.operators.heuristics.large_neighborhood_search._DESTROY_OPS>`
  - ```{autodoc2-docstring} src.policies.other.operators.heuristics.large_neighborhood_search._DESTROY_OPS
    :summary:
    ```
* - {py:obj}`_REPAIR_OPS <src.policies.other.operators.heuristics.large_neighborhood_search._REPAIR_OPS>`
  - ```{autodoc2-docstring} src.policies.other.operators.heuristics.large_neighborhood_search._REPAIR_OPS
    :summary:
    ```
````

### API

````{py:data} _DESTROY_OPS
:canonical: src.policies.other.operators.heuristics.large_neighborhood_search._DESTROY_OPS
:value: >
   None

```{autodoc2-docstring} src.policies.other.operators.heuristics.large_neighborhood_search._DESTROY_OPS
```

````

````{py:data} _REPAIR_OPS
:canonical: src.policies.other.operators.heuristics.large_neighborhood_search._REPAIR_OPS
:value: >
   None

```{autodoc2-docstring} src.policies.other.operators.heuristics.large_neighborhood_search._REPAIR_OPS
```

````

````{py:function} apply_lns(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, rng: random.Random, q: typing.Optional[int] = None, ruin_fraction: typing.Optional[float] = None, destroy_op: str = 'random', repair_op: str = 'greedy', repair_k: int = 2, mandatory_nodes: typing.Optional[typing.List[int]] = None, **kwargs: typing.Any) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.heuristics.large_neighborhood_search.apply_lns

```{autodoc2-docstring} src.policies.other.operators.heuristics.large_neighborhood_search.apply_lns
```
````
