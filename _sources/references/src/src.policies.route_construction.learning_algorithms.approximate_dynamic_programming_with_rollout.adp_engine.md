# {py:mod}`src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine`

```{py:module} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine
```

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ADPRolloutEngine <src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine.ADPRolloutEngine>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine.ADPRolloutEngine
    :summary:
    ```
````

### API

`````{py:class} ADPRolloutEngine(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.params.ADPRolloutParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine.ADPRolloutEngine

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine.ADPRolloutEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine.ADPRolloutEngine.__init__
```

````{py:method} _generate_candidates(wastes: typing.Dict[int, float]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine.ADPRolloutEngine._generate_candidates

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine.ADPRolloutEngine._generate_candidates
```

````

````{py:method} _route_cost(candidate_nodes: typing.List[int], wastes: typing.Dict[int, float]) -> float
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine.ADPRolloutEngine._route_cost

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine.ADPRolloutEngine._route_cost
```

````

````{py:method} _revenue(candidate_nodes: typing.List[int], wastes: typing.Dict[int, float]) -> float
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine.ADPRolloutEngine._revenue

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine.ADPRolloutEngine._revenue
```

````

````{py:method} _simulate_forward(wastes: typing.Dict[int, float], visited_today: typing.Set[int], scenario_tree: typing.Optional[typing.Any], t_current: int, t_total: int) -> float
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine.ADPRolloutEngine._simulate_forward

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine.ADPRolloutEngine._simulate_forward
```

````

````{py:method} solve_day(wastes: typing.Dict[int, float], scenario_tree: typing.Optional[typing.Any], t_current: int, t_total: int) -> typing.Tuple[typing.List[typing.List[int]], float, float, typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine.ADPRolloutEngine.solve_day

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine.ADPRolloutEngine.solve_day
```

````

````{py:method} solve(scenario_tree: typing.Optional[typing.Any], horizon: int) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine.ADPRolloutEngine.solve

```{autodoc2-docstring} src.policies.route_construction.learning_algorithms.approximate_dynamic_programming_with_rollout.adp_engine.ADPRolloutEngine.solve
```

````

`````
