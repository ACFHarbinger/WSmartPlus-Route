# {py:mod}`src.policies.route_improvement.guided_local_search`

```{py:module} src.policies.route_improvement.guided_local_search
```

```{autodoc2-docstring} src.policies.route_improvement.guided_local_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GuidedLocalSearchRouteImprover <src.policies.route_improvement.guided_local_search.GuidedLocalSearchRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.guided_local_search.GuidedLocalSearchRouteImprover
    :summary:
    ```
````

### API

`````{py:class} GuidedLocalSearchRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.guided_local_search.GuidedLocalSearchRouteImprover

Bases: {py:obj}`logic.src.interfaces.route_improvement.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.guided_local_search.GuidedLocalSearchRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.guided_local_search.GuidedLocalSearchRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.policies.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.guided_local_search.GuidedLocalSearchRouteImprover.process

```{autodoc2-docstring} src.policies.route_improvement.guided_local_search.GuidedLocalSearchRouteImprover.process
```

````

````{py:method} _get_operator_method(manager: typing.Any, name: str)
:canonical: src.policies.route_improvement.guided_local_search.GuidedLocalSearchRouteImprover._get_operator_method

```{autodoc2-docstring} src.policies.route_improvement.guided_local_search.GuidedLocalSearchRouteImprover._get_operator_method
```

````

````{py:method} _update_penalties(routes: typing.List[typing.List[int]], dm: numpy.ndarray, penalty: numpy.ndarray)
:canonical: src.policies.route_improvement.guided_local_search.GuidedLocalSearchRouteImprover._update_penalties

```{autodoc2-docstring} src.policies.route_improvement.guided_local_search.GuidedLocalSearchRouteImprover._update_penalties
```

````

`````
