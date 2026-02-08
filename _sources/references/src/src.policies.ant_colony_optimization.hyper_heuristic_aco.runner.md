# {py:mod}`src.policies.ant_colony_optimization.hyper_heuristic_aco.runner`

```{py:module} src.policies.ant_colony_optimization.hyper_heuristic_aco.runner
```

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.runner
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_hyper_heuristic_aco <src.policies.ant_colony_optimization.hyper_heuristic_aco.runner.run_hyper_heuristic_aco>`
  - ```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.runner.run_hyper_heuristic_aco
    :summary:
    ```
* - {py:obj}`_build_greedy_solution <src.policies.ant_colony_optimization.hyper_heuristic_aco.runner._build_greedy_solution>`
  - ```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.runner._build_greedy_solution
    :summary:
    ```
````

### API

````{py:function} run_hyper_heuristic_aco(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, R: float, C: float, values: typing.Dict[str, typing.Any], *args: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.runner.run_hyper_heuristic_aco

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.runner.run_hyper_heuristic_aco
```
````

````{py:function} _build_greedy_solution(nodes: typing.List[int], dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float) -> typing.List[typing.List[int]]
:canonical: src.policies.ant_colony_optimization.hyper_heuristic_aco.runner._build_greedy_solution

```{autodoc2-docstring} src.policies.ant_colony_optimization.hyper_heuristic_aco.runner._build_greedy_solution
```
````
