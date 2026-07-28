# {py:mod}`src.policies.helpers.operators.solution_initialization.regret_si`

```{py:module} src.policies.helpers.operators.solution_initialization.regret_si
```

```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.regret_si
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_best_insertion_cost <src.policies.helpers.operators.solution_initialization.regret_si._best_insertion_cost>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.regret_si._best_insertion_cost
    :summary:
    ```
* - {py:obj}`build_regret_routes <src.policies.helpers.operators.solution_initialization.regret_si.build_regret_routes>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.regret_si.build_regret_routes
    :summary:
    ```
````

### API

````{py:function} _best_insertion_cost(node: int, routes: typing.List[typing.List[int]], loads: typing.List[float], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float) -> typing.Tuple[float, int, int]
:canonical: src.policies.helpers.operators.solution_initialization.regret_si._best_insertion_cost

```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.regret_si._best_insertion_cost
```
````

````{py:function} build_regret_routes(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, regret_k: int = 2, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.solution_initialization.regret_si.build_regret_routes

```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.regret_si.build_regret_routes
```
````
