# {py:mod}`src.policies.route_improvement.multi_phase`

```{py:module} src.policies.route_improvement.multi_phase
```

```{autodoc2-docstring} src.policies.route_improvement.multi_phase
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MultiPhaseRouteImprover <src.policies.route_improvement.multi_phase.MultiPhaseRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.multi_phase.MultiPhaseRouteImprover
    :summary:
    ```
````

### API

`````{py:class} MultiPhaseRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.multi_phase.MultiPhaseRouteImprover

Bases: {py:obj}`logic.src.interfaces.route_improvement.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.multi_phase.MultiPhaseRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.multi_phase.MultiPhaseRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.policies.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.multi_phase.MultiPhaseRouteImprover.process

```{autodoc2-docstring} src.policies.route_improvement.multi_phase.MultiPhaseRouteImprover.process
```

````

`````
