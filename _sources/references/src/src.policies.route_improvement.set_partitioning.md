# {py:mod}`src.policies.route_improvement.set_partitioning`

```{py:module} src.policies.route_improvement.set_partitioning
```

```{autodoc2-docstring} src.policies.route_improvement.set_partitioning
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SetPartitioningRouteImprover <src.policies.route_improvement.set_partitioning.SetPartitioningRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.set_partitioning.SetPartitioningRouteImprover
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_canonical <src.policies.route_improvement.set_partitioning._canonical>`
  - ```{autodoc2-docstring} src.policies.route_improvement.set_partitioning._canonical
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_improvement.set_partitioning.logger>`
  - ```{autodoc2-docstring} src.policies.route_improvement.set_partitioning.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_improvement.set_partitioning.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_improvement.set_partitioning.logger
```

````

````{py:function} _canonical(route: typing.List[int]) -> typing.Tuple[int, ...]
:canonical: src.policies.route_improvement.set_partitioning._canonical

```{autodoc2-docstring} src.policies.route_improvement.set_partitioning._canonical
```
````

`````{py:class} SetPartitioningRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.set_partitioning.SetPartitioningRouteImprover

Bases: {py:obj}`logic.src.interfaces.route_improvement.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.set_partitioning.SetPartitioningRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.set_partitioning.SetPartitioningRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.set_partitioning.SetPartitioningRouteImprover.process

```{autodoc2-docstring} src.policies.route_improvement.set_partitioning.SetPartitioningRouteImprover.process
```

````

`````
