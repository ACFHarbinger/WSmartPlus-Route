# {py:mod}`src.policies.route_improvement.cheapest_insertion`

```{py:module} src.policies.route_improvement.cheapest_insertion
```

```{autodoc2-docstring} src.policies.route_improvement.cheapest_insertion
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CheapestInsertionRouteImprover <src.policies.route_improvement.cheapest_insertion.CheapestInsertionRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.cheapest_insertion.CheapestInsertionRouteImprover
    :summary:
    ```
````

### API

`````{py:class} CheapestInsertionRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.cheapest_insertion.CheapestInsertionRouteImprover

Bases: {py:obj}`logic.src.interfaces.route_improvement.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.cheapest_insertion.CheapestInsertionRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.cheapest_insertion.CheapestInsertionRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.policies.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.cheapest_insertion.CheapestInsertionRouteImprover.process

```{autodoc2-docstring} src.policies.route_improvement.cheapest_insertion.CheapestInsertionRouteImprover.process
```

````

`````
