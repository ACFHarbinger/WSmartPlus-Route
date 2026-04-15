# {py:mod}`src.policies.exact_stochastic_dynamic_programming.esdp_engine`

```{py:module} src.policies.exact_stochastic_dynamic_programming.esdp_engine
```

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.esdp_engine
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ExactSDPEngine <src.policies.exact_stochastic_dynamic_programming.esdp_engine.ExactSDPEngine>`
  - ```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.esdp_engine.ExactSDPEngine
    :summary:
    ```
````

### API

`````{py:class} ExactSDPEngine(params: src.policies.exact_stochastic_dynamic_programming.params.SDPParams, dist_matrix: numpy.ndarray, capacity: float)
:canonical: src.policies.exact_stochastic_dynamic_programming.esdp_engine.ExactSDPEngine

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.esdp_engine.ExactSDPEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.esdp_engine.ExactSDPEngine.__init__
```

````{py:method} _compute_overflow_penalty(state: typing.Tuple[int, ...]) -> float
:canonical: src.policies.exact_stochastic_dynamic_programming.esdp_engine.ExactSDPEngine._compute_overflow_penalty

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.esdp_engine.ExactSDPEngine._compute_overflow_penalty
```

````

````{py:method} _evaluate_action(state: typing.Tuple[int, ...], action: typing.FrozenSet[int], next_val_table: typing.Dict[typing.Tuple[int, ...], float]) -> float
:canonical: src.policies.exact_stochastic_dynamic_programming.esdp_engine.ExactSDPEngine._evaluate_action

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.esdp_engine.ExactSDPEngine._evaluate_action
```

````

````{py:method} compute_overflow_penalty(state: typing.Tuple[int, ...]) -> float
:canonical: src.policies.exact_stochastic_dynamic_programming.esdp_engine.ExactSDPEngine.compute_overflow_penalty

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.esdp_engine.ExactSDPEngine.compute_overflow_penalty
```

````

````{py:method} solve()
:canonical: src.policies.exact_stochastic_dynamic_programming.esdp_engine.ExactSDPEngine.solve

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.esdp_engine.ExactSDPEngine.solve
```

````

````{py:method} get_optimal_action(day: int, state: typing.Tuple[int, ...]) -> typing.FrozenSet[int]
:canonical: src.policies.exact_stochastic_dynamic_programming.esdp_engine.ExactSDPEngine.get_optimal_action

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.esdp_engine.ExactSDPEngine.get_optimal_action
```

````

`````
