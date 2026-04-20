# {py:mod}`src.interfaces.route_improvement`

```{py:module} src.interfaces.route_improvement
```

```{autodoc2-docstring} src.interfaces.route_improvement
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IRouteImprovement <src.interfaces.route_improvement.IRouteImprovement>`
  - ```{autodoc2-docstring} src.interfaces.route_improvement.IRouteImprovement
    :summary:
    ```
````

### API

`````{py:class} IRouteImprovement(**kwargs: typing.Any)
:canonical: src.interfaces.route_improvement.IRouteImprovement

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.interfaces.route_improvement.IRouteImprovement
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.interfaces.route_improvement.IRouteImprovement.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.search_context.ImprovementMetrics]
:canonical: src.interfaces.route_improvement.IRouteImprovement.process
:abstractmethod:

```{autodoc2-docstring} src.interfaces.route_improvement.IRouteImprovement.process
```

````

`````
