# {py:mod}`src.pipeline.rl.meta.td_learning`

```{py:module} src.pipeline.rl.meta.td_learning
```

```{autodoc2-docstring} src.pipeline.rl.meta.td_learning
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CostWeightManager <src.pipeline.rl.meta.td_learning.CostWeightManager>`
  - ```{autodoc2-docstring} src.pipeline.rl.meta.td_learning.CostWeightManager
    :summary:
    ```
````

### API

`````{py:class} CostWeightManager(initial_weights: typing.Optional[typing.Dict[str, float]] = None, learning_rate: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1, n_bins: int = 20, weight_bounds: typing.Optional[typing.Dict[str, typing.Tuple[float, float]]] = None, **kwargs)
:canonical: src.pipeline.rl.meta.td_learning.CostWeightManager

Bases: {py:obj}`logic.src.pipeline.rl.meta.weight_strategy.WeightAdjustmentStrategy`

```{autodoc2-docstring} src.pipeline.rl.meta.td_learning.CostWeightManager
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.meta.td_learning.CostWeightManager.__init__
```

````{py:method} get_current_weights() -> typing.Dict[str, float]
:canonical: src.pipeline.rl.meta.td_learning.CostWeightManager.get_current_weights

```{autodoc2-docstring} src.pipeline.rl.meta.td_learning.CostWeightManager.get_current_weights
```

````

````{py:method} propose_weights(context=None) -> typing.Dict[str, float]
:canonical: src.pipeline.rl.meta.td_learning.CostWeightManager.propose_weights

```{autodoc2-docstring} src.pipeline.rl.meta.td_learning.CostWeightManager.propose_weights
```

````

````{py:method} feedback(reward: float, metrics: typing.List = None, day: int = None)
:canonical: src.pipeline.rl.meta.td_learning.CostWeightManager.feedback

```{autodoc2-docstring} src.pipeline.rl.meta.td_learning.CostWeightManager.feedback
```

````

````{py:method} update_weights(reward, cost_components=None)
:canonical: src.pipeline.rl.meta.td_learning.CostWeightManager.update_weights

```{autodoc2-docstring} src.pipeline.rl.meta.td_learning.CostWeightManager.update_weights
```

````

````{py:method} _discretize(weights: typing.Dict[str, float]) -> typing.Tuple[int, ...]
:canonical: src.pipeline.rl.meta.td_learning.CostWeightManager._discretize

```{autodoc2-docstring} src.pipeline.rl.meta.td_learning.CostWeightManager._discretize
```

````

````{py:method} _apply_change(key: str, delta: float)
:canonical: src.pipeline.rl.meta.td_learning.CostWeightManager._apply_change

```{autodoc2-docstring} src.pipeline.rl.meta.td_learning.CostWeightManager._apply_change
```

````

````{py:method} state_dict()
:canonical: src.pipeline.rl.meta.td_learning.CostWeightManager.state_dict

```{autodoc2-docstring} src.pipeline.rl.meta.td_learning.CostWeightManager.state_dict
```

````

````{py:method} load_state_dict(state_dict)
:canonical: src.pipeline.rl.meta.td_learning.CostWeightManager.load_state_dict

```{autodoc2-docstring} src.pipeline.rl.meta.td_learning.CostWeightManager.load_state_dict
```

````

`````
