# {py:mod}`src.policies.route_improvement.fix_and_optimize`

```{py:module} src.policies.route_improvement.fix_and_optimize
```

```{autodoc2-docstring} src.policies.route_improvement.fix_and_optimize
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FixAndOptimizeRouteImprover <src.policies.route_improvement.fix_and_optimize.FixAndOptimizeRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.fix_and_optimize.FixAndOptimizeRouteImprover
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_improvement.fix_and_optimize.logger>`
  - ```{autodoc2-docstring} src.policies.route_improvement.fix_and_optimize.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_improvement.fix_and_optimize.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_improvement.fix_and_optimize.logger
```

````

`````{py:class} FixAndOptimizeRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.fix_and_optimize.FixAndOptimizeRouteImprover

Bases: {py:obj}`logic.src.interfaces.route_improvement.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.fix_and_optimize.FixAndOptimizeRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.fix_and_optimize.FixAndOptimizeRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.policies.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.fix_and_optimize.FixAndOptimizeRouteImprover.process

````

`````
