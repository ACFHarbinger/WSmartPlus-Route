# {py:mod}`src.policies.lin_kernighan_helsgaun_three.load_tracker`

```{py:module} src.policies.lin_kernighan_helsgaun_three.load_tracker
```

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LoadState <src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`build_load_state <src.policies.lin_kernighan_helsgaun_three.load_tracker.build_load_state>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.build_load_state
    :summary:
    ```
* - {py:obj}`get_route_nodes <src.policies.lin_kernighan_helsgaun_three.load_tracker.get_route_nodes>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.get_route_nodes
    :summary:
    ```
* - {py:obj}`calculate_route_penalty <src.policies.lin_kernighan_helsgaun_three.load_tracker.calculate_route_penalty>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.calculate_route_penalty
    :summary:
    ```
* - {py:obj}`get_exact_penalty_delta <src.policies.lin_kernighan_helsgaun_three.load_tracker.get_exact_penalty_delta>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.get_exact_penalty_delta
    :summary:
    ```
* - {py:obj}`calculate_penalty_delta_exact <src.policies.lin_kernighan_helsgaun_three.load_tracker.calculate_penalty_delta_exact>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.calculate_penalty_delta_exact
    :summary:
    ```
* - {py:obj}`update_load_state_after_move <src.policies.lin_kernighan_helsgaun_three.load_tracker.update_load_state_after_move>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.update_load_state_after_move
    :summary:
    ```
* - {py:obj}`get_affected_route_indices <src.policies.lin_kernighan_helsgaun_three.load_tracker.get_affected_route_indices>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.get_affected_route_indices
    :summary:
    ```
````

### API

`````{py:class} LoadState(route_assignments: typing.Dict[int, int], route_loads: typing.Dict[int, float], route_penalties: typing.Dict[int, float], capacity: float, n_routes: int)
:canonical: src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState.__init__
```

````{py:method} get_total_penalty() -> float
:canonical: src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState.get_total_penalty

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState.get_total_penalty
```

````

````{py:method} get_route_for_node(node: int) -> typing.Optional[int]
:canonical: src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState.get_route_for_node

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState.get_route_for_node
```

````

````{py:method} copy() -> src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState
:canonical: src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState.copy

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState.copy
```

````

`````

````{py:function} build_load_state(tour: typing.List[int], waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float], n_original: int) -> typing.Optional[src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState]
:canonical: src.policies.lin_kernighan_helsgaun_three.load_tracker.build_load_state

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.build_load_state
```
````

````{py:function} get_route_nodes(tour: typing.List[int], route_idx: int, n_original: int) -> typing.List[int]
:canonical: src.policies.lin_kernighan_helsgaun_three.load_tracker.get_route_nodes

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.get_route_nodes
```
````

````{py:function} calculate_route_penalty(nodes: typing.List[int], waste: numpy.ndarray, capacity: float) -> typing.Tuple[float, float]
:canonical: src.policies.lin_kernighan_helsgaun_three.load_tracker.calculate_route_penalty

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.calculate_route_penalty
```
````

````{py:function} get_exact_penalty_delta(curr_tour: typing.List[int], broken_edges: typing.List[typing.Tuple[int, int]], added_edges: typing.List[typing.Tuple[int, int]], state: src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState, waste: numpy.ndarray, capacity: float, n_original: typing.Optional[int] = None) -> float
:canonical: src.policies.lin_kernighan_helsgaun_three.load_tracker.get_exact_penalty_delta

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.get_exact_penalty_delta
```
````

````{py:function} calculate_penalty_delta_exact(old_tour: typing.List[int], new_tour: typing.List[int], waste: numpy.ndarray, capacity: float, n_original: int) -> float
:canonical: src.policies.lin_kernighan_helsgaun_three.load_tracker.calculate_penalty_delta_exact

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.calculate_penalty_delta_exact
```
````

````{py:function} update_load_state_after_move(state: src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState, new_tour: typing.List[int], waste: numpy.ndarray, capacity: float, n_original: int) -> src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState
:canonical: src.policies.lin_kernighan_helsgaun_three.load_tracker.update_load_state_after_move

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.update_load_state_after_move
```
````

````{py:function} get_affected_route_indices(edges: typing.List[typing.Tuple[int, int]], state: src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState) -> set
:canonical: src.policies.lin_kernighan_helsgaun_three.load_tracker.get_affected_route_indices

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.load_tracker.get_affected_route_indices
```
````
