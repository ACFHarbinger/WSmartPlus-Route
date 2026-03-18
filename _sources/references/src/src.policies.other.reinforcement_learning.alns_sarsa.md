# {py:mod}`src.policies.other.reinforcement_learning.alns_sarsa`

```{py:module} src.policies.other.reinforcement_learning.alns_sarsa
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ALNSSARSASolver <src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver>`
  - ```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver
    :summary:
    ```
````

### API

`````{py:class} ALNSSARSASolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Any, rl_params: typing.Any, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None, evaluator=None)
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver.__init__
```

````{py:method} _init_destroy_operators()
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._init_destroy_operators

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._init_destroy_operators
```

````

````{py:method} _init_repair_operators()
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._init_repair_operators

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._init_repair_operators
```

````

````{py:method} _init_perturbation_operators()
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._init_perturbation_operators

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._init_perturbation_operators
```

````

````{py:method} _destroy_random(routes: typing.List[typing.List[int]], n: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._destroy_random

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._destroy_random
```

````

````{py:method} _destroy_worst(routes: typing.List[typing.List[int]], n: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._destroy_worst

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._destroy_worst
```

````

````{py:method} _destroy_cluster(routes: typing.List[typing.List[int]], n: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._destroy_cluster

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._destroy_cluster
```

````

````{py:method} _destroy_shaw(routes: typing.List[typing.List[int]], n: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._destroy_shaw

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._destroy_shaw
```

````

````{py:method} _destroy_string(routes: typing.List[typing.List[int]], n: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._destroy_string

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._destroy_string
```

````

````{py:method} _repair_greedy(routes: typing.List[typing.List[int]], removed: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._repair_greedy

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._repair_greedy
```

````

````{py:method} _repair_regret2(routes: typing.List[typing.List[int]], removed: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._repair_regret2

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._repair_regret2
```

````

````{py:method} _repair_regretk(routes: typing.List[typing.List[int]], removed: typing.List[int], k: int) -> typing.List[typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._repair_regretk

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._repair_regretk
```

````

````{py:method} _repair_greedy_blink(routes: typing.List[typing.List[int]], removed: typing.List[int], blink_rate: float = 0.1) -> typing.List[typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._repair_greedy_blink

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._repair_greedy_blink
```

````

````{py:method} _repair_string_type_i(routes: typing.List[typing.List[int]], removed: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._repair_string_type_i

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._repair_string_type_i
```

````

````{py:method} _repair_string_type_ii(routes: typing.List[typing.List[int]], removed: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._repair_string_type_ii

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._repair_string_type_ii
```

````

````{py:method} _repair_string_type_iii(routes: typing.List[typing.List[int]], removed: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._repair_string_type_iii

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._repair_string_type_iii
```

````

````{py:method} _repair_string_type_iv(routes: typing.List[typing.List[int]], removed: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._repair_string_type_iv

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._repair_string_type_iv
```

````

````{py:method} _unstring_wrapper(routes: typing.List[typing.List[int]], n: int, op_type: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._unstring_wrapper

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._unstring_wrapper
```

````

````{py:method} _unstring_type_i(routes: typing.List[typing.List[int]], n: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._unstring_type_i

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._unstring_type_i
```

````

````{py:method} _unstring_type_ii(routes: typing.List[typing.List[int]], n: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._unstring_type_ii

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._unstring_type_ii
```

````

````{py:method} _unstring_type_iii(routes: typing.List[typing.List[int]], n: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._unstring_type_iii

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._unstring_type_iii
```

````

````{py:method} _unstring_type_iv(routes: typing.List[typing.List[int]], n: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._unstring_type_iv

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._unstring_type_iv
```

````

````{py:method} _perturb_kick(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._perturb_kick

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._perturb_kick
```

````

````{py:method} _perturb_random(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._perturb_random

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._perturb_random
```

````

````{py:method} solve(initial_solution: typing.Optional[typing.List[typing.List[int]]] = None) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver.solve

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver.solve
```

````

````{py:method} _initialize_solve(initial_solution: typing.Optional[typing.List[typing.List[int]]]) -> typing.Tuple[typing.List[typing.List[int]], typing.List[typing.List[int]], float, float]
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._initialize_solve

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._initialize_solve
```

````

````{py:method} _calc_removal_size(routes: typing.List[typing.List[int]]) -> int
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._calc_removal_size

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._calc_removal_size
```

````

````{py:method} _calculate_diversity(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._calculate_diversity

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver._calculate_diversity
```

````

````{py:method} calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver.calculate_cost

```{autodoc2-docstring} src.policies.other.reinforcement_learning.alns_sarsa.ALNSSARSASolver.calculate_cost
```

````

`````
