# {py:mod}`src.policies.other.post_processing.learned`

```{py:module} src.policies.other.post_processing.learned
```

```{autodoc2-docstring} src.policies.other.post_processing.learned
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MoveScorer <src.policies.other.post_processing.learned.MoveScorer>`
  - ```{autodoc2-docstring} src.policies.other.post_processing.learned.MoveScorer
    :summary:
    ```
* - {py:obj}`LearnedPostProcessor <src.policies.other.post_processing.learned.LearnedPostProcessor>`
  - ```{autodoc2-docstring} src.policies.other.post_processing.learned.LearnedPostProcessor
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.other.post_processing.learned.logger>`
  - ```{autodoc2-docstring} src.policies.other.post_processing.learned.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.other.post_processing.learned.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.other.post_processing.learned.logger
```

````

`````{py:class} MoveScorer(node_dim: int = 5, edge_dim: int = 2, hidden_dim: int = 64)
:canonical: src.policies.other.post_processing.learned.MoveScorer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.policies.other.post_processing.learned.MoveScorer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.post_processing.learned.MoveScorer.__init__
```

````{py:method} forward(node_features: torch.Tensor, move_endpoints: torch.Tensor, move_types: torch.Tensor) -> torch.Tensor
:canonical: src.policies.other.post_processing.learned.MoveScorer.forward

```{autodoc2-docstring} src.policies.other.post_processing.learned.MoveScorer.forward
```

````

`````

`````{py:class} LearnedPostProcessor(**kwargs: typing.Any)
:canonical: src.policies.other.post_processing.learned.LearnedPostProcessor

Bases: {py:obj}`logic.src.interfaces.post_processing.IPostProcessor`

```{autodoc2-docstring} src.policies.other.post_processing.learned.LearnedPostProcessor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.post_processing.learned.LearnedPostProcessor.__init__
```

````{py:attribute} DEFAULT_WEIGHTS_PATH
:canonical: src.policies.other.post_processing.learned.LearnedPostProcessor.DEFAULT_WEIGHTS_PATH
:value: >
   None

```{autodoc2-docstring} src.policies.other.post_processing.learned.LearnedPostProcessor.DEFAULT_WEIGHTS_PATH
```

````

````{py:method} _lazy_load_model(weights_path: pathlib.Path) -> None
:canonical: src.policies.other.post_processing.learned.LearnedPostProcessor._lazy_load_model

```{autodoc2-docstring} src.policies.other.post_processing.learned.LearnedPostProcessor._lazy_load_model
```

````

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.List[int]
:canonical: src.policies.other.post_processing.learned.LearnedPostProcessor.process

````

````{py:method} _apply_learned_moves(routes: typing.List[typing.List[int]], dm: numpy.ndarray, wastes: dict, capacity: float, mandatory_nodes: set, max_iterations: int, min_improvement: float, neighborhood_size: int) -> typing.List[typing.List[int]]
:canonical: src.policies.other.post_processing.learned.LearnedPostProcessor._apply_learned_moves

```{autodoc2-docstring} src.policies.other.post_processing.learned.LearnedPostProcessor._apply_learned_moves
```

````

````{py:method} _enumerate_moves(routes: typing.List[typing.List[int]], dm: numpy.ndarray, k: int) -> typing.List[typing.Tuple[str, int, int, int, int]]
:canonical: src.policies.other.post_processing.learned.LearnedPostProcessor._enumerate_moves

```{autodoc2-docstring} src.policies.other.post_processing.learned.LearnedPostProcessor._enumerate_moves
```

````

````{py:method} _build_node_features(routes: typing.List[typing.List[int]], dm: numpy.ndarray, wastes: dict, mandatory_nodes: set) -> torch.Tensor
:canonical: src.policies.other.post_processing.learned.LearnedPostProcessor._build_node_features

```{autodoc2-docstring} src.policies.other.post_processing.learned.LearnedPostProcessor._build_node_features
```

````

````{py:method} _apply_move(routes: typing.List[typing.List[int]], move: typing.Tuple[str, int, int, int, int]) -> typing.List[typing.List[int]]
:canonical: src.policies.other.post_processing.learned.LearnedPostProcessor._apply_move

```{autodoc2-docstring} src.policies.other.post_processing.learned.LearnedPostProcessor._apply_move
```

````

````{py:method} _fallback(tour: typing.List[int], dm: numpy.ndarray, wastes: dict, capacity: float, cost_per_km: float, revenue_kg: float, **kwargs: typing.Any) -> typing.List[int]
:canonical: src.policies.other.post_processing.learned.LearnedPostProcessor._fallback

```{autodoc2-docstring} src.policies.other.post_processing.learned.LearnedPostProcessor._fallback
```

````

`````
