# {py:mod}`src.policies.route_improvement.learned`

```{py:module} src.policies.route_improvement.learned
```

```{autodoc2-docstring} src.policies.route_improvement.learned
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MoveScorer <src.policies.route_improvement.learned.MoveScorer>`
  - ```{autodoc2-docstring} src.policies.route_improvement.learned.MoveScorer
    :summary:
    ```
* - {py:obj}`LearnedRouteImprover <src.policies.route_improvement.learned.LearnedRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.learned.LearnedRouteImprover
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_improvement.learned.logger>`
  - ```{autodoc2-docstring} src.policies.route_improvement.learned.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_improvement.learned.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_improvement.learned.logger
```

````

`````{py:class} MoveScorer(node_dim: int = 5, edge_dim: int = 2, hidden_dim: int = 64)
:canonical: src.policies.route_improvement.learned.MoveScorer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.policies.route_improvement.learned.MoveScorer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.learned.MoveScorer.__init__
```

````{py:method} forward(node_features: torch.Tensor, move_endpoints: torch.Tensor, move_types: torch.Tensor) -> torch.Tensor
:canonical: src.policies.route_improvement.learned.MoveScorer.forward

```{autodoc2-docstring} src.policies.route_improvement.learned.MoveScorer.forward
```

````

`````

`````{py:class} LearnedRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.learned.LearnedRouteImprover

Bases: {py:obj}`logic.src.interfaces.route_improvement.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.learned.LearnedRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.learned.LearnedRouteImprover.__init__
```

````{py:attribute} DEFAULT_WEIGHTS_PATH
:canonical: src.policies.route_improvement.learned.LearnedRouteImprover.DEFAULT_WEIGHTS_PATH
:value: >
   None

```{autodoc2-docstring} src.policies.route_improvement.learned.LearnedRouteImprover.DEFAULT_WEIGHTS_PATH
```

````

````{py:method} _lazy_load_model(weights_path: pathlib.Path) -> None
:canonical: src.policies.route_improvement.learned.LearnedRouteImprover._lazy_load_model

```{autodoc2-docstring} src.policies.route_improvement.learned.LearnedRouteImprover._lazy_load_model
```

````

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.policies.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.learned.LearnedRouteImprover.process

````

````{py:method} _apply_learned_moves(routes: typing.List[typing.List[int]], dm: numpy.ndarray, wastes: dict, capacity: float, mandatory_nodes: set, max_iterations: int, min_improvement: float, neighborhood_size: int) -> typing.List[typing.List[int]]
:canonical: src.policies.route_improvement.learned.LearnedRouteImprover._apply_learned_moves

```{autodoc2-docstring} src.policies.route_improvement.learned.LearnedRouteImprover._apply_learned_moves
```

````

````{py:method} _enumerate_moves(routes: typing.List[typing.List[int]], dm: numpy.ndarray, k: int) -> typing.List[typing.Tuple[str, int, int, int, int]]
:canonical: src.policies.route_improvement.learned.LearnedRouteImprover._enumerate_moves

```{autodoc2-docstring} src.policies.route_improvement.learned.LearnedRouteImprover._enumerate_moves
```

````

````{py:method} _build_node_features(routes: typing.List[typing.List[int]], dm: numpy.ndarray, wastes: dict, mandatory_nodes: set) -> torch.Tensor
:canonical: src.policies.route_improvement.learned.LearnedRouteImprover._build_node_features

```{autodoc2-docstring} src.policies.route_improvement.learned.LearnedRouteImprover._build_node_features
```

````

````{py:method} _apply_move(routes: typing.List[typing.List[int]], move: typing.Tuple[str, int, int, int, int]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_improvement.learned.LearnedRouteImprover._apply_move

```{autodoc2-docstring} src.policies.route_improvement.learned.LearnedRouteImprover._apply_move
```

````

````{py:method} _fallback(tour: typing.List[int], dm: numpy.ndarray, wastes: dict, capacity: float, cost_per_km: float, revenue_kg: float, **kwargs: typing.Any) -> typing.List[int]
:canonical: src.policies.route_improvement.learned.LearnedRouteImprover._fallback

```{autodoc2-docstring} src.policies.route_improvement.learned.LearnedRouteImprover._fallback
```

````

`````
