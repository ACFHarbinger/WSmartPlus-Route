# {py:mod}`src.policies.other.reinforcement_learning.features.state`

```{py:module} src.policies.other.reinforcement_learning.features.state
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.features.state
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StateFeatureExtractor <src.policies.other.reinforcement_learning.features.state.StateFeatureExtractor>`
  - ```{autodoc2-docstring} src.policies.other.reinforcement_learning.features.state.StateFeatureExtractor
    :summary:
    ```
````

### API

`````{py:class} StateFeatureExtractor(progress_thresholds: typing.Optional[typing.Tuple[float, float]] = None, stagnation_thresholds: typing.Optional[typing.Tuple[int, int]] = None, diversity_thresholds: typing.Optional[typing.Tuple[float, float]] = None)
:canonical: src.policies.other.reinforcement_learning.features.state.StateFeatureExtractor

```{autodoc2-docstring} src.policies.other.reinforcement_learning.features.state.StateFeatureExtractor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.features.state.StateFeatureExtractor.__init__
```

````{py:method} extract_features(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, iteration: int, max_iterations: int, current_cost: float, best_cost: float, stagnation_count: int, improvement_history: typing.List[float]) -> typing.Dict[str, float]
:canonical: src.policies.other.reinforcement_learning.features.state.StateFeatureExtractor.extract_features

```{autodoc2-docstring} src.policies.other.reinforcement_learning.features.state.StateFeatureExtractor.extract_features
```

````

````{py:method} discretize_state(iteration: int, max_iterations: int, stagnation_count: int, diversity: float) -> typing.Tuple[int, int, int]
:canonical: src.policies.other.reinforcement_learning.features.state.StateFeatureExtractor.discretize_state

```{autodoc2-docstring} src.policies.other.reinforcement_learning.features.state.StateFeatureExtractor.discretize_state
```

````

````{py:method} state_to_index(state_tuple: typing.Tuple[int, int, int]) -> int
:canonical: src.policies.other.reinforcement_learning.features.state.StateFeatureExtractor.state_to_index

```{autodoc2-docstring} src.policies.other.reinforcement_learning.features.state.StateFeatureExtractor.state_to_index
```

````

`````
