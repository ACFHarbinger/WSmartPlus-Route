# {py:mod}`src.policies.route_improvement.dp_route_reopt`

```{py:module} src.policies.route_improvement.dp_route_reopt
```

```{autodoc2-docstring} src.policies.route_improvement.dp_route_reopt
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DPRouteReoptRouteImprover <src.policies.route_improvement.dp_route_reopt.DPRouteReoptRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.dp_route_reopt.DPRouteReoptRouteImprover
    :summary:
    ```
````

### API

`````{py:class} DPRouteReoptRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.dp_route_reopt.DPRouteReoptRouteImprover

Bases: {py:obj}`logic.src.interfaces.route_improvement.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.dp_route_reopt.DPRouteReoptRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.dp_route_reopt.DPRouteReoptRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.policies.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.dp_route_reopt.DPRouteReoptRouteImprover.process

````

`````
