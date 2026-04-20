# {py:mod}`src.policies.route_improvement.adaptive_ensemble`

```{py:module} src.policies.route_improvement.adaptive_ensemble
```

```{autodoc2-docstring} src.policies.route_improvement.adaptive_ensemble
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AdaptiveEnsembleRouteImprover <src.policies.route_improvement.adaptive_ensemble.AdaptiveEnsembleRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.adaptive_ensemble.AdaptiveEnsembleRouteImprover
    :summary:
    ```
````

### API

`````{py:class} AdaptiveEnsembleRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.adaptive_ensemble.AdaptiveEnsembleRouteImprover

Bases: {py:obj}`logic.src.interfaces.route_improvement.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.adaptive_ensemble.AdaptiveEnsembleRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.adaptive_ensemble.AdaptiveEnsembleRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.adaptive_ensemble.AdaptiveEnsembleRouteImprover.process

````

`````
