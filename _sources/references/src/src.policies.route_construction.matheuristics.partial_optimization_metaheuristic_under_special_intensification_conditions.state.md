# {py:mod}`src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state`

```{py:module} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`POPMUSICState <src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState
    :summary:
    ```
````

### API

`````{py:class} POPMUSICState(n_nodes: int, coord_array: numpy.ndarray)
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState.__init__
```

````{py:method} alloc_slot() -> int
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState.alloc_slot

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState.alloc_slot
```

````

````{py:method} insert_route(nodes: typing.List[int], centroid: numpy.ndarray) -> int
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState.insert_route

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState.insert_route
```

````

````{py:method} insert_singleton(node: int) -> int
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState.insert_singleton

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState.insert_singleton
```

````

````{py:method} delete_slot(slot: int) -> None
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState.delete_slot

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState.delete_slot
```

````

````{py:method} update_coords(slot: int, centroid: numpy.ndarray) -> None
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState.update_coords

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState.update_coords
```

````

````{py:method} customer_nodes(slot: int) -> typing.List[int]
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState.customer_nodes

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState.customer_nodes
```

````

````{py:property} active_route_slots
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState.active_route_slots
:type: typing.Set[int]

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState.active_route_slots
```

````

````{py:property} active_singleton_slots
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState.active_singleton_slots
:type: typing.Set[int]

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState.active_singleton_slots
```

````

`````
