# {py:mod}`src.policies.cluster_first_route_second.greedy_assignment`

```{py:module} src.policies.cluster_first_route_second.greedy_assignment
```

```{autodoc2-docstring} src.policies.cluster_first_route_second.greedy_assignment
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`assign_greedy <src.policies.cluster_first_route_second.greedy_assignment.assign_greedy>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.greedy_assignment.assign_greedy
    :summary:
    ```
* - {py:obj}`_handle_unassigned_nodes <src.policies.cluster_first_route_second.greedy_assignment._handle_unassigned_nodes>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.greedy_assignment._handle_unassigned_nodes
    :summary:
    ```
* - {py:obj}`_greedy_swap_fallback <src.policies.cluster_first_route_second.greedy_assignment._greedy_swap_fallback>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.greedy_assignment._greedy_swap_fallback
    :summary:
    ```
````

### API

````{py:function} assign_greedy(seeds: typing.List[int], must_go: typing.List[int], wastes: typing.Dict[int, float], capacity: float, distance_matrix: numpy.ndarray, strict_fleet: bool = False) -> typing.List[typing.List[int]]
:canonical: src.policies.cluster_first_route_second.greedy_assignment.assign_greedy

```{autodoc2-docstring} src.policies.cluster_first_route_second.greedy_assignment.assign_greedy
```
````

````{py:function} _handle_unassigned_nodes(unassigned: typing.List[int], must_go: typing.List[int], seeds: typing.List[int], clusters: typing.List[typing.List[int]], loads: typing.List[float], assigned: typing.Set[int], capacity: float, wastes: typing.Dict[int, float], strict_fleet: bool) -> None
:canonical: src.policies.cluster_first_route_second.greedy_assignment._handle_unassigned_nodes

```{autodoc2-docstring} src.policies.cluster_first_route_second.greedy_assignment._handle_unassigned_nodes
```
````

````{py:function} _greedy_swap_fallback(node: int, waste: float, seeds: typing.List[int], clusters: typing.List[typing.List[int]], loads: typing.List[float], assigned: typing.Set[int], capacity: float, wastes: typing.Dict[int, float]) -> bool
:canonical: src.policies.cluster_first_route_second.greedy_assignment._greedy_swap_fallback

```{autodoc2-docstring} src.policies.cluster_first_route_second.greedy_assignment._greedy_swap_fallback
```
````
