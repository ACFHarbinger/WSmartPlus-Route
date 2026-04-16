# {py:mod}`src.interfaces.distance_metric`

```{py:module} src.interfaces.distance_metric
```

```{autodoc2-docstring} src.interfaces.distance_metric
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IDistanceMetric <src.interfaces.distance_metric.IDistanceMetric>`
  - ```{autodoc2-docstring} src.interfaces.distance_metric.IDistanceMetric
    :summary:
    ```
````

### API

`````{py:class} IDistanceMetric
:canonical: src.interfaces.distance_metric.IDistanceMetric

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.interfaces.distance_metric.IDistanceMetric
```

````{py:method} compute(current: typing.Any, candidate: typing.Any) -> float
:canonical: src.interfaces.distance_metric.IDistanceMetric.compute
:abstractmethod:

```{autodoc2-docstring} src.interfaces.distance_metric.IDistanceMetric.compute
```

````

`````
