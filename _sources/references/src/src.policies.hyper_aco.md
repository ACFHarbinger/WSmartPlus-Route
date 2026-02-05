# {py:mod}`src.policies.hyper_aco`

```{py:module} src.policies.hyper_aco
```

```{autodoc2-docstring} src.policies.hyper_aco
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HyperACOParams <src.policies.hyper_aco.HyperACOParams>`
  - ```{autodoc2-docstring} src.policies.hyper_aco.HyperACOParams
    :summary:
    ```
* - {py:obj}`HyperHeuristicACO <src.policies.hyper_aco.HyperHeuristicACO>`
  - ```{autodoc2-docstring} src.policies.hyper_aco.HyperHeuristicACO
    :summary:
    ```
````

### API

`````{py:class} HyperACOParams
:canonical: src.policies.hyper_aco.HyperACOParams

```{autodoc2-docstring} src.policies.hyper_aco.HyperACOParams
```

````{py:attribute} n_ants
:canonical: src.policies.hyper_aco.HyperACOParams.n_ants
:type: int
:value: >
   10

```{autodoc2-docstring} src.policies.hyper_aco.HyperACOParams.n_ants
```

````

````{py:attribute} alpha
:canonical: src.policies.hyper_aco.HyperACOParams.alpha
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.hyper_aco.HyperACOParams.alpha
```

````

````{py:attribute} beta
:canonical: src.policies.hyper_aco.HyperACOParams.beta
:type: float
:value: >
   2.0

```{autodoc2-docstring} src.policies.hyper_aco.HyperACOParams.beta
```

````

````{py:attribute} rho
:canonical: src.policies.hyper_aco.HyperACOParams.rho
:type: float
:value: >
   0.1

```{autodoc2-docstring} src.policies.hyper_aco.HyperACOParams.rho
```

````

````{py:attribute} tau_0
:canonical: src.policies.hyper_aco.HyperACOParams.tau_0
:type: float
:value: >
   1.0

```{autodoc2-docstring} src.policies.hyper_aco.HyperACOParams.tau_0
```

````

````{py:attribute} tau_min
:canonical: src.policies.hyper_aco.HyperACOParams.tau_min
:type: float
:value: >
   0.01

```{autodoc2-docstring} src.policies.hyper_aco.HyperACOParams.tau_min
```

````

````{py:attribute} tau_max
:canonical: src.policies.hyper_aco.HyperACOParams.tau_max
:type: float
:value: >
   10.0

```{autodoc2-docstring} src.policies.hyper_aco.HyperACOParams.tau_max
```

````

````{py:attribute} max_iterations
:canonical: src.policies.hyper_aco.HyperACOParams.max_iterations
:type: int
:value: >
   50

```{autodoc2-docstring} src.policies.hyper_aco.HyperACOParams.max_iterations
```

````

````{py:attribute} time_limit
:canonical: src.policies.hyper_aco.HyperACOParams.time_limit
:type: float
:value: >
   30.0

```{autodoc2-docstring} src.policies.hyper_aco.HyperACOParams.time_limit
```

````

````{py:attribute} sequence_length
:canonical: src.policies.hyper_aco.HyperACOParams.sequence_length
:type: int
:value: >
   5

```{autodoc2-docstring} src.policies.hyper_aco.HyperACOParams.sequence_length
```

````

````{py:attribute} q0
:canonical: src.policies.hyper_aco.HyperACOParams.q0
:type: float
:value: >
   0.9

```{autodoc2-docstring} src.policies.hyper_aco.HyperACOParams.q0
```

````

````{py:attribute} operators
:canonical: src.policies.hyper_aco.HyperACOParams.operators
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.hyper_aco.HyperACOParams.operators
```

````

`````

`````{py:class} HyperHeuristicACO(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, C: float, params: typing.Optional[src.policies.hyper_aco.HyperACOParams] = None)
:canonical: src.policies.hyper_aco.HyperHeuristicACO

```{autodoc2-docstring} src.policies.hyper_aco.HyperHeuristicACO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hyper_aco.HyperHeuristicACO.__init__
```

````{py:method} _init_pheromones()
:canonical: src.policies.hyper_aco.HyperHeuristicACO._init_pheromones

```{autodoc2-docstring} src.policies.hyper_aco.HyperHeuristicACO._init_pheromones
```

````

````{py:method} solve(initial_routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.hyper_aco.HyperHeuristicACO.solve

```{autodoc2-docstring} src.policies.hyper_aco.HyperHeuristicACO.solve
```

````

````{py:method} _construct_sequence() -> typing.List[str]
:canonical: src.policies.hyper_aco.HyperHeuristicACO._construct_sequence

```{autodoc2-docstring} src.policies.hyper_aco.HyperHeuristicACO._construct_sequence
```

````

````{py:method} _select_next_operator(current: int) -> int
:canonical: src.policies.hyper_aco.HyperHeuristicACO._select_next_operator

```{autodoc2-docstring} src.policies.hyper_aco.HyperHeuristicACO._select_next_operator
```

````

````{py:method} _apply_sequence(routes: typing.List[typing.List[int]], sequence: typing.List[str]) -> typing.List[typing.List[int]]
:canonical: src.policies.hyper_aco.HyperHeuristicACO._apply_sequence

```{autodoc2-docstring} src.policies.hyper_aco.HyperHeuristicACO._apply_sequence
```

````

````{py:method} _calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.hyper_aco.HyperHeuristicACO._calculate_cost

```{autodoc2-docstring} src.policies.hyper_aco.HyperHeuristicACO._calculate_cost
```

````

````{py:method} _evaporate_pheromones()
:canonical: src.policies.hyper_aco.HyperHeuristicACO._evaporate_pheromones

```{autodoc2-docstring} src.policies.hyper_aco.HyperHeuristicACO._evaporate_pheromones
```

````

````{py:method} _deposit_pheromones(solutions: typing.List[typing.Tuple[typing.List[typing.List[int]], float, typing.List[str]]])
:canonical: src.policies.hyper_aco.HyperHeuristicACO._deposit_pheromones

```{autodoc2-docstring} src.policies.hyper_aco.HyperHeuristicACO._deposit_pheromones
```

````

````{py:method} _update_heuristics()
:canonical: src.policies.hyper_aco.HyperHeuristicACO._update_heuristics

```{autodoc2-docstring} src.policies.hyper_aco.HyperHeuristicACO._update_heuristics
```

````

`````
