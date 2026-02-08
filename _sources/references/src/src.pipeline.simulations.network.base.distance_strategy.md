# {py:mod}`src.pipeline.simulations.network.base.distance_strategy`

```{py:module} src.pipeline.simulations.network.base.distance_strategy
```

```{autodoc2-docstring} src.pipeline.simulations.network.base.distance_strategy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DistanceStrategy <src.pipeline.simulations.network.base.distance_strategy.DistanceStrategy>`
  - ```{autodoc2-docstring} src.pipeline.simulations.network.base.distance_strategy.DistanceStrategy
    :summary:
    ```
````

### API

`````{py:class} DistanceStrategy
:canonical: src.pipeline.simulations.network.base.distance_strategy.DistanceStrategy

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.pipeline.simulations.network.base.distance_strategy.DistanceStrategy
```

````{py:method} calculate(coords: pandas.DataFrame, **kwargs: typing.Any) -> numpy.ndarray
:canonical: src.pipeline.simulations.network.base.distance_strategy.DistanceStrategy.calculate
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.network.base.distance_strategy.DistanceStrategy.calculate
```

````

````{py:method} _eval_kwarg(kwarg: str, kwargs: typing.Dict[str, typing.Any]) -> bool
:canonical: src.pipeline.simulations.network.base.distance_strategy.DistanceStrategy._eval_kwarg

```{autodoc2-docstring} src.pipeline.simulations.network.base.distance_strategy.DistanceStrategy._eval_kwarg
```

````

`````
