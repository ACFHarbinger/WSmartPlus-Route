# {py:mod}`src.pipeline.simulations.network.base.iterative_distance_streategy`

```{py:module} src.pipeline.simulations.network.base.iterative_distance_streategy
```

```{autodoc2-docstring} src.pipeline.simulations.network.base.iterative_distance_streategy
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IterativeDistanceStrategy <src.pipeline.simulations.network.base.iterative_distance_streategy.IterativeDistanceStrategy>`
  - ```{autodoc2-docstring} src.pipeline.simulations.network.base.iterative_distance_streategy.IterativeDistanceStrategy
    :summary:
    ```
````

### API

`````{py:class} IterativeDistanceStrategy
:canonical: src.pipeline.simulations.network.base.iterative_distance_streategy.IterativeDistanceStrategy

Bases: {py:obj}`src.pipeline.simulations.network.base.distance_strategy.DistanceStrategy`

```{autodoc2-docstring} src.pipeline.simulations.network.base.iterative_distance_streategy.IterativeDistanceStrategy
```

````{py:method} calculate_pair(coords_i: typing.Tuple[float, float], coords_j: typing.Tuple[float, float]) -> float
:canonical: src.pipeline.simulations.network.base.iterative_distance_streategy.IterativeDistanceStrategy.calculate_pair
:abstractmethod:

```{autodoc2-docstring} src.pipeline.simulations.network.base.iterative_distance_streategy.IterativeDistanceStrategy.calculate_pair
```

````

````{py:method} calculate(coords: pandas.DataFrame, **kwargs: typing.Any) -> numpy.ndarray
:canonical: src.pipeline.simulations.network.base.iterative_distance_streategy.IterativeDistanceStrategy.calculate

```{autodoc2-docstring} src.pipeline.simulations.network.base.iterative_distance_streategy.IterativeDistanceStrategy.calculate
```

````

`````
