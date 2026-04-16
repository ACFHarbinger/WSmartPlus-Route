# {py:mod}`src.policies.route_improvement.steepest_two_opt`

```{py:module} src.policies.route_improvement.steepest_two_opt
```

```{autodoc2-docstring} src.policies.route_improvement.steepest_two_opt
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SteepestTwoOptRouteImprover <src.policies.route_improvement.steepest_two_opt.SteepestTwoOptRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.steepest_two_opt.SteepestTwoOptRouteImprover
    :summary:
    ```
````

### API

`````{py:class} SteepestTwoOptRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.steepest_two_opt.SteepestTwoOptRouteImprover

Bases: {py:obj}`logic.src.interfaces.route_improvement.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.steepest_two_opt.SteepestTwoOptRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.steepest_two_opt.SteepestTwoOptRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.policies.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.steepest_two_opt.SteepestTwoOptRouteImprover.process

````

`````
