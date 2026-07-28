# {py:mod}`src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators`

```{py:module} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_handle_2opt <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_2opt>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_2opt
    :summary:
    ```
* - {py:obj}`_handle_move <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_move>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_move
    :summary:
    ```
* - {py:obj}`_handle_swap <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_swap>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_swap
    :summary:
    ```
* - {py:obj}`_handle_insert <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_insert>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_insert
    :summary:
    ```
* - {py:obj}`apply_operator <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators.apply_operator>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators.apply_operator
    :summary:
    ```
````

### API

````{py:function} _handle_2opt(solution: typing.List[typing.List[int]], rng: random.Random) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_2opt

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_2opt
```
````

````{py:function} _handle_move(solution: typing.List[typing.List[int]], data: typing.Dict[str, typing.Any], vehicle_capacity: float, id_to_index: typing.Dict[int, int], rng: random.Random) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_move

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_move
```
````

````{py:function} _handle_swap(solution, rng)
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_swap

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_swap
```
````

````{py:function} _handle_insert(solution: typing.List[typing.List[int]], data: typing.Dict[str, typing.Any], stocks: typing.Dict[int, float], vehicle_capacity: float, id_to_index: typing.Dict[int, int], distance_matrix: numpy.ndarray, candidate_removed_bins: typing.Set[int], rng: random.Random) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_insert

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators._handle_insert
```
````

````{py:function} apply_operator(op: str, new_solution: typing.List[typing.List[int]], candidate_removed_bins: typing.Set[int], data: dict, vehicle_capacity: float, id_to_index: dict, stocks: typing.Dict[int, float], mandatory_bins: typing.Set[int], distance_matrix: numpy.ndarray, rng: random.Random) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators.apply_operator

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators.apply_operator
```
````
