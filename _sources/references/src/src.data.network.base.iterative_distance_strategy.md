# {py:mod}`src.data.network.base.iterative_distance_strategy`

```{py:module} src.data.network.base.iterative_distance_strategy
```

```{autodoc2-docstring} src.data.network.base.iterative_distance_strategy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IterativeDistanceStrategy <src.data.network.base.iterative_distance_strategy.IterativeDistanceStrategy>`
  - ```{autodoc2-docstring} src.data.network.base.iterative_distance_strategy.IterativeDistanceStrategy
    :summary:
    ```
````

### API

`````{py:class} IterativeDistanceStrategy
:canonical: src.data.network.base.iterative_distance_strategy.IterativeDistanceStrategy

Bases: {py:obj}`src.data.network.base.distance_strategy.DistanceStrategy`

```{autodoc2-docstring} src.data.network.base.iterative_distance_strategy.IterativeDistanceStrategy
```

````{py:method} calculate_pair(coords_i: typing.Tuple[float, float], coords_j: typing.Tuple[float, float]) -> float
:canonical: src.data.network.base.iterative_distance_strategy.IterativeDistanceStrategy.calculate_pair
:abstractmethod:

```{autodoc2-docstring} src.data.network.base.iterative_distance_strategy.IterativeDistanceStrategy.calculate_pair
```

````

````{py:method} calculate(coords: pandas.DataFrame, **kwargs: typing.Any) -> numpy.ndarray
:canonical: src.data.network.base.iterative_distance_strategy.IterativeDistanceStrategy.calculate

```{autodoc2-docstring} src.data.network.base.iterative_distance_strategy.IterativeDistanceStrategy.calculate
```

````

`````
