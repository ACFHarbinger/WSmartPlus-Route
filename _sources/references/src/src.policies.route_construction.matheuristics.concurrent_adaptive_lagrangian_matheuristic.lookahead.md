# {py:mod}`src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead`

```{py:module} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LookaheadTables <src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadTables>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadTables
    :summary:
    ```
* - {py:obj}`LookaheadValuator <src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadValuator>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadValuator
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`lagrangian_corrected_prizes <src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.lagrangian_corrected_prizes>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.lagrangian_corrected_prizes
    :summary:
    ```
````

### API

`````{py:class} LookaheadTables
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadTables

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadTables
```

````{py:attribute} V
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadTables.V
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadTables.V
```

````

````{py:attribute} rho
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadTables.rho
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadTables.rho
```

````

````{py:attribute} early_regret
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadTables.early_regret
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadTables.early_regret
```

````

````{py:attribute} expected_fill
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadTables.expected_fill
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadTables.expected_fill
```

````

`````

`````{py:class} LookaheadValuator(params: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.params.LookaheadParams)
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadValuator

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadValuator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadValuator.__init__
```

````{py:method} get_scenario_tree() -> logic.src.pipeline.simulations.bins.prediction.ScenarioTree
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadValuator.get_scenario_tree

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadValuator.get_scenario_tree
```

````

````{py:method} compute(current_wastes: numpy.ndarray, bin_stats: typing.Optional[typing.Dict[str, numpy.ndarray]] = None, truth_generator: typing.Optional[object] = None) -> src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadTables
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadValuator.compute

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadValuator.compute
```

````

````{py:method} _extract_expected_path(tree: logic.src.pipeline.simulations.bins.prediction.ScenarioTree, N: int) -> numpy.ndarray
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadValuator._extract_expected_path

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadValuator._extract_expected_path
```

````

````{py:method} _compute_value_table(expected_fill: numpy.ndarray) -> numpy.ndarray
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadValuator._compute_value_table

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadValuator._compute_value_table
```

````

````{py:method} _compute_regret_table(V: numpy.ndarray) -> numpy.ndarray
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadValuator._compute_regret_table

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.LookaheadValuator._compute_regret_table
```

````

`````

````{py:function} lagrangian_corrected_prizes(V: numpy.ndarray, lambdas: numpy.ndarray, insertion_costs: numpy.ndarray, gamma: float, side: str = 'knapsack') -> numpy.ndarray
:canonical: src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.lagrangian_corrected_prizes

```{autodoc2-docstring} src.policies.route_construction.matheuristics.concurrent_adaptive_lagrangian_matheuristic.lookahead.lagrangian_corrected_prizes
```
````
