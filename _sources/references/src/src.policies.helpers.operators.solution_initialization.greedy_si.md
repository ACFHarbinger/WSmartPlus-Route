# {py:mod}`src.policies.helpers.operators.solution_initialization.greedy_si`

```{py:module} src.policies.helpers.operators.solution_initialization.greedy_si
```

```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.greedy_si
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_greedy_profit_insertion <src.policies.helpers.operators.solution_initialization.greedy_si._greedy_profit_insertion>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.greedy_si._greedy_profit_insertion
    :summary:
    ```
* - {py:obj}`build_greedy_routes <src.policies.helpers.operators.solution_initialization.greedy_si.build_greedy_routes>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.greedy_si.build_greedy_routes
    :summary:
    ```
````

### API

````{py:function} _greedy_profit_insertion(routes: typing.List[typing.List[int]], unvisited_optional: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes_set: set[int], rng: random.Random) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.solution_initialization.greedy_si._greedy_profit_insertion

```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.greedy_si._greedy_profit_insertion
```
````

````{py:function} build_greedy_routes(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.solution_initialization.greedy_si.build_greedy_routes

```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.greedy_si.build_greedy_routes
```
````
