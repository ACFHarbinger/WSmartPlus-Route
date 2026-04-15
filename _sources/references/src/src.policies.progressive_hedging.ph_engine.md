# {py:mod}`src.policies.progressive_hedging.ph_engine`

```{py:module} src.policies.progressive_hedging.ph_engine
```

```{autodoc2-docstring} src.policies.progressive_hedging.ph_engine
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ProgressiveHedgingEngine <src.policies.progressive_hedging.ph_engine.ProgressiveHedgingEngine>`
  - ```{autodoc2-docstring} src.policies.progressive_hedging.ph_engine.ProgressiveHedgingEngine
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.progressive_hedging.ph_engine.logger>`
  - ```{autodoc2-docstring} src.policies.progressive_hedging.ph_engine.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.progressive_hedging.ph_engine.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.progressive_hedging.ph_engine.logger
```

````

`````{py:class} ProgressiveHedgingEngine(config: logic.src.configs.policies.PHConfig)
:canonical: src.policies.progressive_hedging.ph_engine.ProgressiveHedgingEngine

```{autodoc2-docstring} src.policies.progressive_hedging.ph_engine.ProgressiveHedgingEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.progressive_hedging.ph_engine.ProgressiveHedgingEngine.__init__
```

````{py:method} ensure_route_list(routes: typing.Union[typing.List[int], typing.List[typing.List[int]]]) -> typing.List[typing.List[int]]
:canonical: src.policies.progressive_hedging.ph_engine.ProgressiveHedgingEngine.ensure_route_list
:staticmethod:

```{autodoc2-docstring} src.policies.progressive_hedging.ph_engine.ProgressiveHedgingEngine.ensure_route_list
```

````

````{py:method} solve(sub_dist_matrix: numpy.ndarray, scenario_wastes: typing.List[typing.Dict[int, float]], capacity: float, revenue: float, cost_unit: float, mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.progressive_hedging.ph_engine.ProgressiveHedgingEngine.solve

```{autodoc2-docstring} src.policies.progressive_hedging.ph_engine.ProgressiveHedgingEngine.solve
```

````

`````
