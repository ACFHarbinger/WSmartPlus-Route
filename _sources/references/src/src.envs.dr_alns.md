# {py:mod}`src.envs.dr_alns`

```{py:module} src.envs.dr_alns
```

```{autodoc2-docstring} src.envs.dr_alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DRALNSEnv <src.envs.dr_alns.DRALNSEnv>`
  - ```{autodoc2-docstring} src.envs.dr_alns.DRALNSEnv
    :summary:
    ```
````

### API

`````{py:class} DRALNSEnv(max_iterations: int = 100, n_destroy_ops: int = 3, n_repair_ops: int = 2, instance_generator: typing.Optional[typing.Any] = None, seed: typing.Optional[int] = None)
:canonical: src.envs.dr_alns.DRALNSEnv

Bases: {py:obj}`gymnasium.Env`

```{autodoc2-docstring} src.envs.dr_alns.DRALNSEnv
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.dr_alns.DRALNSEnv.__init__
```

````{py:attribute} metadata
:canonical: src.envs.dr_alns.DRALNSEnv.metadata
:value: >
   None

```{autodoc2-docstring} src.envs.dr_alns.DRALNSEnv.metadata
```

````

````{py:method} reset(seed: typing.Optional[int] = None, options: typing.Optional[typing.Dict[str, typing.Any]] = None) -> typing.Tuple[numpy.ndarray, typing.Dict[str, typing.Any]]
:canonical: src.envs.dr_alns.DRALNSEnv.reset

```{autodoc2-docstring} src.envs.dr_alns.DRALNSEnv.reset
```

````

````{py:method} step(action: numpy.ndarray) -> typing.Tuple[numpy.ndarray, float, bool, bool, typing.Dict[str, typing.Any]]
:canonical: src.envs.dr_alns.DRALNSEnv.step

```{autodoc2-docstring} src.envs.dr_alns.DRALNSEnv.step
```

````

````{py:method} _get_observation() -> numpy.ndarray
:canonical: src.envs.dr_alns.DRALNSEnv._get_observation

```{autodoc2-docstring} src.envs.dr_alns.DRALNSEnv._get_observation
```

````

````{py:method} _apply_operators(routes: typing.List[typing.List[int]], destroy_idx: int, repair_idx: int, severity: float) -> typing.List[typing.List[int]]
:canonical: src.envs.dr_alns.DRALNSEnv._apply_operators

```{autodoc2-docstring} src.envs.dr_alns.DRALNSEnv._apply_operators
```

````

````{py:method} _accept(current_profit: float, new_profit: float, temperature: float) -> bool
:canonical: src.envs.dr_alns.DRALNSEnv._accept

```{autodoc2-docstring} src.envs.dr_alns.DRALNSEnv._accept
```

````

````{py:method} _random_removal(routes: typing.List[typing.List[int]], n: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.envs.dr_alns.DRALNSEnv._random_removal

```{autodoc2-docstring} src.envs.dr_alns.DRALNSEnv._random_removal
```

````

````{py:method} _worst_removal(routes: typing.List[typing.List[int]], n: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.envs.dr_alns.DRALNSEnv._worst_removal

```{autodoc2-docstring} src.envs.dr_alns.DRALNSEnv._worst_removal
```

````

````{py:method} _cluster_removal(routes: typing.List[typing.List[int]], n: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.envs.dr_alns.DRALNSEnv._cluster_removal

```{autodoc2-docstring} src.envs.dr_alns.DRALNSEnv._cluster_removal
```

````

````{py:method} _greedy_insertion(partial_routes: typing.List[typing.List[int]], removed: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.envs.dr_alns.DRALNSEnv._greedy_insertion

```{autodoc2-docstring} src.envs.dr_alns.DRALNSEnv._greedy_insertion
```

````

````{py:method} _regret_2_insertion(partial_routes: typing.List[typing.List[int]], removed: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.envs.dr_alns.DRALNSEnv._regret_2_insertion

```{autodoc2-docstring} src.envs.dr_alns.DRALNSEnv._regret_2_insertion
```

````

````{py:method} _build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.envs.dr_alns.DRALNSEnv._build_initial_solution

```{autodoc2-docstring} src.envs.dr_alns.DRALNSEnv._build_initial_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.envs.dr_alns.DRALNSEnv._evaluate

```{autodoc2-docstring} src.envs.dr_alns.DRALNSEnv._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.envs.dr_alns.DRALNSEnv._cost

```{autodoc2-docstring} src.envs.dr_alns.DRALNSEnv._cost
```

````

`````
