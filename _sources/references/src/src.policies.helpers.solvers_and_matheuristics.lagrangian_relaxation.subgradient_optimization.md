# {py:mod}`src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.subgradient_optimization`

```{py:module} src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.subgradient_optimization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.subgradient_optimization
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_nearest_neighbour_tour_cost <src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.subgradient_optimization._nearest_neighbour_tour_cost>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.subgradient_optimization._nearest_neighbour_tour_cost
    :summary:
    ```
* - {py:obj}`run_subgradient <src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.subgradient_optimization.run_subgradient>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.subgradient_optimization.run_subgradient
    :summary:
    ```
````

### API

````{py:function} _nearest_neighbour_tour_cost(visited: typing.Set[int], wastes: typing.Dict[int, float], dist_matrix: numpy.ndarray, R: float, C: float) -> float
:canonical: src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.subgradient_optimization._nearest_neighbour_tour_cost

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.subgradient_optimization._nearest_neighbour_tour_cost
```
````

````{py:function} run_subgradient(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_indices: typing.Optional[typing.Set[int]] = None, params: typing.Optional[typing.Any] = None, time_budget: float = 60.0, env: typing.Any = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None) -> typing.Tuple[float, float, float, typing.List[typing.Dict[str, float]]]
:canonical: src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.subgradient_optimization.run_subgradient

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.subgradient_optimization.run_subgradient
```
````
