# {py:mod}`src.policies.route_improvement.set_partitioning_polish`

```{py:module} src.policies.route_improvement.set_partitioning_polish
```

```{autodoc2-docstring} src.policies.route_improvement.set_partitioning_polish
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SetPartitioningPolishRouteImprover <src.policies.route_improvement.set_partitioning_polish.SetPartitioningPolishRouteImprover>`
  - ```{autodoc2-docstring} src.policies.route_improvement.set_partitioning_polish.SetPartitioningPolishRouteImprover
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_improvement.set_partitioning_polish.logger>`
  - ```{autodoc2-docstring} src.policies.route_improvement.set_partitioning_polish.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_improvement.set_partitioning_polish.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_improvement.set_partitioning_polish.logger
```

````

`````{py:class} SetPartitioningPolishRouteImprover(**kwargs: typing.Any)
:canonical: src.policies.route_improvement.set_partitioning_polish.SetPartitioningPolishRouteImprover

Bases: {py:obj}`logic.src.interfaces.route_improvement.IRouteImprovement`

```{autodoc2-docstring} src.policies.route_improvement.set_partitioning_polish.SetPartitioningPolishRouteImprover
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_improvement.set_partitioning_polish.SetPartitioningPolishRouteImprover.__init__
```

````{py:method} process(tour: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[int], logic.src.interfaces.context.search_context.ImprovementMetrics]
:canonical: src.policies.route_improvement.set_partitioning_polish.SetPartitioningPolishRouteImprover.process

````

`````
