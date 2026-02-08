# {py:mod}`src.policies.simulated_annealing_neighborhood_search.refinement.rebalancing`

```{py:module} src.policies.simulated_annealing_neighborhood_search.refinement.rebalancing
```

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.refinement.rebalancing
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`rebalance_solution <src.policies.simulated_annealing_neighborhood_search.refinement.rebalancing.rebalance_solution>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.refinement.rebalancing.rebalance_solution
    :summary:
    ```
````

### API

````{py:function} rebalance_solution(solution: typing.List[typing.List[int]], removed_bins: typing.List[int], p_vehicle: float, p_load: float, p_route_difference: float, p_shift: float, data: pandas.DataFrame, must_go_bins: typing.List[int], distance_matrix: numpy.ndarray, values: typing.Dict, iterations: int = 10) -> typing.Tuple[typing.List[typing.List[int]], float, typing.List[int]]
:canonical: src.policies.simulated_annealing_neighborhood_search.refinement.rebalancing.rebalance_solution

```{autodoc2-docstring} src.policies.simulated_annealing_neighborhood_search.refinement.rebalancing.rebalance_solution
```
````
