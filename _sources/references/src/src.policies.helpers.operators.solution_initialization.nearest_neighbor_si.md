# {py:mod}`src.policies.helpers.operators.solution_initialization.nearest_neighbor_si`

```{py:module} src.policies.helpers.operators.solution_initialization.nearest_neighbor_si
```

```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.nearest_neighbor_si
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`build_nn_routes <src.policies.helpers.operators.solution_initialization.nearest_neighbor_si.build_nn_routes>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.nearest_neighbor_si.build_nn_routes
    :summary:
    ```
````

### API

````{py:function} build_nn_routes(nodes: typing.List[int], mandatory_nodes: typing.List[int], wastes: typing.Dict[int, float], capacity: float, dist_matrix: numpy.ndarray, R: float, C: float, rng: typing.Optional[random.Random] = None, prune_unprofitable: bool = True) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.solution_initialization.nearest_neighbor_si.build_nn_routes

```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.nearest_neighbor_si.build_nn_routes
```
````
