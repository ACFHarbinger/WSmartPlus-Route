# {py:mod}`src.policies.route_improvement.mip_lns`

```{py:module} src.policies.route_improvement.mip_lns
```

```{autodoc2-docstring} src.policies.route_improvement.mip_lns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MIPLNSRouteImprover <src.policies.route_improvement.mip_lns.MIPLNSRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.mip_lns.MIPLNSRouteImprover
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_improvement.mip_lns.logger>`
  - ```{autodoc2-docstring} src.policies.route_improvement.mip_lns.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_improvement.mip_lns.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_improvement.mip_lns.logger
```

````

`````{py:class} MIPLNSRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.mip_lns.MIPLNSRouteImprover

Bases: {py:obj}`logic.src.interfaces.route_improvement.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.mip_lns.MIPLNSRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.mip_lns.MIPLNSRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.mip_lns.MIPLNSRouteImprover.process

```{autodoc2-docstring} src.policies.route_improvement.mip_lns.MIPLNSRouteImprover.process
```

````

`````
