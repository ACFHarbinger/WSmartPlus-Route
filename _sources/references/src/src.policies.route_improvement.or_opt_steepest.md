# {py:mod}`src.policies.route_improvement.or_opt_steepest`

```{py:module} src.policies.route_improvement.or_opt_steepest
```

```{autodoc2-docstring} src.policies.route_improvement.or_opt_steepest
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OrOptSteepestRouteImprover <src.policies.route_improvement.or_opt_steepest.OrOptSteepestRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.or_opt_steepest.OrOptSteepestRouteImprover
    :summary:
    ```
````

### API

`````{py:class} OrOptSteepestRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.or_opt_steepest.OrOptSteepestRouteImprover

Bases: {py:obj}`logic.src.interfaces.route_improvement.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.or_opt_steepest.OrOptSteepestRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.or_opt_steepest.OrOptSteepestRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.or_opt_steepest.OrOptSteepestRouteImprover.process

````

`````
