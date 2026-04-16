# {py:mod}`src.policies.route_improvement.path`

```{py:module} src.policies.route_improvement.path
```

```{autodoc2-docstring} src.policies.route_improvement.path
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PathRouteImprover <src.policies.route_improvement.path.PathRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.path.PathRouteImprover
    :summary:
    ```
````

### API

`````{py:class} PathRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.path.PathRouteImprover

Bases: {py:obj}`logic.src.interfaces.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.path.PathRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.path.PathRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.policies.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.path.PathRouteImprover.process

```{autodoc2-docstring} src.policies.route_improvement.path.PathRouteImprover.process
```

````

`````
