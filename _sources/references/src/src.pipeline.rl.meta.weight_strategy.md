# {py:mod}`src.pipeline.rl.meta.weight_strategy`

```{py:module} src.pipeline.rl.meta.weight_strategy
```

```{autodoc2-docstring} src.pipeline.rl.meta.weight_strategy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`WeightAdjustmentStrategy <src.pipeline.rl.meta.weight_strategy.WeightAdjustmentStrategy>`
  - ```{autodoc2-docstring} src.pipeline.rl.meta.weight_strategy.WeightAdjustmentStrategy
    :summary:
    ```
````

### API

`````{py:class} WeightAdjustmentStrategy
:canonical: src.pipeline.rl.meta.weight_strategy.WeightAdjustmentStrategy

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.pipeline.rl.meta.weight_strategy.WeightAdjustmentStrategy
```

````{py:method} propose_weights(context: typing.Optional[typing.Dict[str, typing.Any]] = None) -> typing.Dict[str, float]
:canonical: src.pipeline.rl.meta.weight_strategy.WeightAdjustmentStrategy.propose_weights
:abstractmethod:

```{autodoc2-docstring} src.pipeline.rl.meta.weight_strategy.WeightAdjustmentStrategy.propose_weights
```

````

````{py:method} feedback(reward: float, metrics: typing.Any, day: typing.Optional[int] = None, step: typing.Optional[int] = None)
:canonical: src.pipeline.rl.meta.weight_strategy.WeightAdjustmentStrategy.feedback
:abstractmethod:

```{autodoc2-docstring} src.pipeline.rl.meta.weight_strategy.WeightAdjustmentStrategy.feedback
```

````

````{py:method} get_current_weights() -> typing.Dict[str, float]
:canonical: src.pipeline.rl.meta.weight_strategy.WeightAdjustmentStrategy.get_current_weights
:abstractmethod:

```{autodoc2-docstring} src.pipeline.rl.meta.weight_strategy.WeightAdjustmentStrategy.get_current_weights
```

````

`````
