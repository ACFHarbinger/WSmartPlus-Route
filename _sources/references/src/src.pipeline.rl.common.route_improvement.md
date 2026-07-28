# {py:mod}`src.pipeline.rl.common.route_improvement`

```{py:module} src.pipeline.rl.common.route_improvement
```

```{autodoc2-docstring} src.pipeline.rl.common.route_improvement
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EfficiencyOptimizer <src.pipeline.rl.common.route_improvement.EfficiencyOptimizer>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.route_improvement.EfficiencyOptimizer
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`decode_routes <src.pipeline.rl.common.route_improvement.decode_routes>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.route_improvement.decode_routes
    :summary:
    ```
* - {py:obj}`calculate_efficiency <src.pipeline.rl.common.route_improvement.calculate_efficiency>`
  - ```{autodoc2-docstring} src.pipeline.rl.common.route_improvement.calculate_efficiency
    :summary:
    ```
````

### API

`````{py:class} EfficiencyOptimizer(problem, **kwargs)
:canonical: src.pipeline.rl.common.route_improvement.EfficiencyOptimizer

```{autodoc2-docstring} src.pipeline.rl.common.route_improvement.EfficiencyOptimizer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.common.route_improvement.EfficiencyOptimizer.__init__
```

````{py:method} optimize(routes: typing.List[torch.Tensor], **kwargs)
:canonical: src.pipeline.rl.common.route_improvement.EfficiencyOptimizer.optimize

```{autodoc2-docstring} src.pipeline.rl.common.route_improvement.EfficiencyOptimizer.optimize
```

````

`````

````{py:function} decode_routes(actions: torch.Tensor, num_nodes: int) -> typing.List[typing.List[int]]
:canonical: src.pipeline.rl.common.route_improvement.decode_routes

```{autodoc2-docstring} src.pipeline.rl.common.route_improvement.decode_routes
```
````

````{py:function} calculate_efficiency(routes, dist_matrix, waste, capacity)
:canonical: src.pipeline.rl.common.route_improvement.calculate_efficiency

```{autodoc2-docstring} src.pipeline.rl.common.route_improvement.calculate_efficiency
```
````
