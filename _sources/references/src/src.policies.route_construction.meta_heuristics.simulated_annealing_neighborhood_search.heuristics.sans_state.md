# {py:mod}`src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_state`

```{py:module} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_state
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_state
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_profit <src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_state.compute_profit>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_state.compute_profit
    :summary:
    ```
````

### API

````{py:function} compute_profit(solution: typing.List[typing.List[int]], distance_matrix: typing.List[typing.List[float]], id_to_index: typing.Dict[int, int], data: typing.Dict[str, typing.Any], vehicle_capacity: float, R: float, V: float, density: float, mandatory_bins: typing.Set[int], stocks: typing.Optional[typing.Dict[int, float]] = None) -> typing.Tuple[float, float, float, float]
:canonical: src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_state.compute_profit

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_state.compute_profit
```
````
