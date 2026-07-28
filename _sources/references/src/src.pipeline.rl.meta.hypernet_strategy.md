# {py:mod}`src.pipeline.rl.meta.hypernet_strategy`

```{py:module} src.pipeline.rl.meta.hypernet_strategy
```

```{autodoc2-docstring} src.pipeline.rl.meta.hypernet_strategy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HyperNetworkStrategy <src.pipeline.rl.meta.hypernet_strategy.HyperNetworkStrategy>`
  - ```{autodoc2-docstring} src.pipeline.rl.meta.hypernet_strategy.HyperNetworkStrategy
    :summary:
    ```
````

### API

`````{py:class} HyperNetworkStrategy(problem: typing.Any, device: typing.Any, initial_weights: typing.Dict[str, float], lr: float = 0.0001, constraint_value: float = 1.0, buffer_size: int = 100, **kwargs)
:canonical: src.pipeline.rl.meta.hypernet_strategy.HyperNetworkStrategy

Bases: {py:obj}`logic.src.pipeline.rl.meta.weight_strategy.WeightAdjustmentStrategy`

```{autodoc2-docstring} src.pipeline.rl.meta.hypernet_strategy.HyperNetworkStrategy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.meta.hypernet_strategy.HyperNetworkStrategy.__init__
```

````{py:method} propose_weights(context: typing.Optional[typing.Dict[str, typing.Any]] = None) -> typing.Dict[str, float]
:canonical: src.pipeline.rl.meta.hypernet_strategy.HyperNetworkStrategy.propose_weights

```{autodoc2-docstring} src.pipeline.rl.meta.hypernet_strategy.HyperNetworkStrategy.propose_weights
```

````

````{py:method} feedback(reward: float, metrics: typing.Any, day: typing.Optional[int] = None, step: typing.Optional[int] = None)
:canonical: src.pipeline.rl.meta.hypernet_strategy.HyperNetworkStrategy.feedback

```{autodoc2-docstring} src.pipeline.rl.meta.hypernet_strategy.HyperNetworkStrategy.feedback
```

````

````{py:method} get_current_weights() -> typing.Dict[str, float]
:canonical: src.pipeline.rl.meta.hypernet_strategy.HyperNetworkStrategy.get_current_weights

```{autodoc2-docstring} src.pipeline.rl.meta.hypernet_strategy.HyperNetworkStrategy.get_current_weights
```

````

`````
