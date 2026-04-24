# {py:mod}`src.policies.route_improvement.lkh`

```{py:module} src.policies.route_improvement.lkh
```

```{autodoc2-docstring} src.policies.route_improvement.lkh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LKHRouteImprover <src.policies.route_improvement.lkh.LKHRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.lkh.LKHRouteImprover
    :summary:
    ```
````

### API

`````{py:class} LKHRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.lkh.LKHRouteImprover

Bases: {py:obj}`logic.src.interfaces.route_improvement.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.lkh.LKHRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.lkh.LKHRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.lkh.LKHRouteImprover.process

```{autodoc2-docstring} src.policies.route_improvement.lkh.LKHRouteImprover.process
```

````

`````
