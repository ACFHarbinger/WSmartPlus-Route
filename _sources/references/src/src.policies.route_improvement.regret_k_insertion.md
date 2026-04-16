# {py:mod}`src.policies.route_improvement.regret_k_insertion`

```{py:module} src.policies.route_improvement.regret_k_insertion
```

```{autodoc2-docstring} src.policies.route_improvement.regret_k_insertion
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RegretKInsertionRouteImprover <src.policies.route_improvement.regret_k_insertion.RegretKInsertionRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.regret_k_insertion.RegretKInsertionRouteImprover
    :summary:
    ```
````

### API

`````{py:class} RegretKInsertionRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.regret_k_insertion.RegretKInsertionRouteImprover

Bases: {py:obj}`logic.src.interfaces.route_improvement.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.regret_k_insertion.RegretKInsertionRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.regret_k_insertion.RegretKInsertionRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.policies.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.regret_k_insertion.RegretKInsertionRouteImprover.process

```{autodoc2-docstring} src.policies.route_improvement.regret_k_insertion.RegretKInsertionRouteImprover.process
```

````

`````
