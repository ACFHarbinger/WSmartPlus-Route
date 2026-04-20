# {py:mod}`src.policies.route_improvement.fast_tsp`

```{py:module} src.policies.route_improvement.fast_tsp
```

```{autodoc2-docstring} src.policies.route_improvement.fast_tsp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FastTSPRouteImprover <src.policies.route_improvement.fast_tsp.FastTSPRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.fast_tsp.FastTSPRouteImprover
    :summary:
    ```
````

### API

`````{py:class} FastTSPRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.fast_tsp.FastTSPRouteImprover

Bases: {py:obj}`logic.src.interfaces.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.fast_tsp.FastTSPRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.fast_tsp.FastTSPRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.fast_tsp.FastTSPRouteImprover.process

```{autodoc2-docstring} src.policies.route_improvement.fast_tsp.FastTSPRouteImprover.process
```

````

`````
