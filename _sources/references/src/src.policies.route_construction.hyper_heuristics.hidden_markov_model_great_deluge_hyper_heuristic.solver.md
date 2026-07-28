# {py:mod}`src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver`

```{py:module} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HMMGDHHSolver <src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_OBS_IMPROVE <src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_IMPROVE>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_IMPROVE
    :summary:
    ```
* - {py:obj}`_OBS_WORSE <src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_WORSE>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_WORSE
    :summary:
    ```
* - {py:obj}`_OBS_WORSE_COST <src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_WORSE_COST>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_WORSE_COST
    :summary:
    ```
* - {py:obj}`_OBS_SAME <src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_SAME>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_SAME
    :summary:
    ```
* - {py:obj}`_OBS_SAME_COST <src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_SAME_COST>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_SAME_COST
    :summary:
    ```
* - {py:obj}`_N_OBS <src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._N_OBS>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._N_OBS
    :summary:
    ```
````

### API

````{py:data} _OBS_IMPROVE
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_IMPROVE
:value: >
   0

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_IMPROVE
```

````

````{py:data} _OBS_WORSE
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_WORSE
:value: >
   1

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_WORSE
```

````

````{py:data} _OBS_WORSE_COST
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_WORSE_COST
:value: >
   2

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_WORSE_COST
```

````

````{py:data} _OBS_SAME
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_SAME
:value: >
   3

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_SAME
```

````

````{py:data} _OBS_SAME_COST
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_SAME_COST
:value: >
   4

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._OBS_SAME_COST
```

````

````{py:data} _N_OBS
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._N_OBS
:value: >
   5

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver._N_OBS
```

````

`````{py:class} HMMGDHHSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.params.HMMGDHHParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver.__init__
```

````{py:method} solve(initial_routes: typing.List[typing.List[int]]) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver.solve

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver.solve
```

````

````{py:method} _map_observation(delta_profit: float, delta_cost: float) -> int
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._map_observation

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._map_observation
```

````

````{py:method} _check_state_scaling() -> None
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._check_state_scaling

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._check_state_scaling
```

````

````{py:method} _split_state() -> None
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._split_state

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._split_state
```

````

````{py:method} _online_em_update(u_idx: int, o_idx: int)
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._online_em_update

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._online_em_update
```

````

````{py:method} _select_action(iteration: int, max_iterations: int) -> int
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._select_action

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._select_action
```

````

````{py:method} _llh0(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh0

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh0
```

````

````{py:method} _llh1(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh1

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh1
```

````

````{py:method} _llh2(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh2

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh2
```

````

````{py:method} _llh3(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh3

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh3
```

````

````{py:method} _llh4(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh4

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh4
```

````

````{py:method} _llh5(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh5

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh5
```

````

````{py:method} _llh6(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh6

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._llh6
```

````

````{py:method} _build_random_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._build_random_solution

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._build_random_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._evaluate

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._cost

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic.solver.HMMGDHHSolver._cost
```

````

`````
