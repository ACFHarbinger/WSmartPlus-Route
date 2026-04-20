# {py:mod}`src.policies.route_improvement.local_search`

```{py:module} src.policies.route_improvement.local_search
```

```{autodoc2-docstring} src.policies.route_improvement.local_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ClassicalLocalSearchRouteImprover <src.policies.route_improvement.local_search.ClassicalLocalSearchRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.local_search.ClassicalLocalSearchRouteImprover
    :summary:
    ```
````

### API

`````{py:class} ClassicalLocalSearchRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.local_search.ClassicalLocalSearchRouteImprover

Bases: {py:obj}`logic.src.interfaces.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.local_search.ClassicalLocalSearchRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.local_search.ClassicalLocalSearchRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.local_search.ClassicalLocalSearchRouteImprover.process

```{autodoc2-docstring} src.policies.route_improvement.local_search.ClassicalLocalSearchRouteImprover.process
```

````

`````
