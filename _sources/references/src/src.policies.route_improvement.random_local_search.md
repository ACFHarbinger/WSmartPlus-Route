# {py:mod}`src.policies.route_improvement.random_local_search`

```{py:module} src.policies.route_improvement.random_local_search
```

```{autodoc2-docstring} src.policies.route_improvement.random_local_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RandomLocalSearchRouteImprover <src.policies.route_improvement.random_local_search.RandomLocalSearchRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.random_local_search.RandomLocalSearchRouteImprover
    :summary:
    ```
````

### API

`````{py:class} RandomLocalSearchRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.random_local_search.RandomLocalSearchRouteImprover

Bases: {py:obj}`logic.src.interfaces.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.random_local_search.RandomLocalSearchRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.random_local_search.RandomLocalSearchRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.policies.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.random_local_search.RandomLocalSearchRouteImprover.process

```{autodoc2-docstring} src.policies.route_improvement.random_local_search.RandomLocalSearchRouteImprover.process
```

````

`````
