# {py:mod}`src.pipeline.rl.features.post_processing`

```{py:module} src.pipeline.rl.features.post_processing
```

```{autodoc2-docstring} src.pipeline.rl.features.post_processing
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EfficiencyOptimizer <src.pipeline.rl.features.post_processing.EfficiencyOptimizer>`
  - ```{autodoc2-docstring} src.pipeline.rl.features.post_processing.EfficiencyOptimizer
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`decode_routes <src.pipeline.rl.features.post_processing.decode_routes>`
  - ```{autodoc2-docstring} src.pipeline.rl.features.post_processing.decode_routes
    :summary:
    ```
* - {py:obj}`calculate_efficiency <src.pipeline.rl.features.post_processing.calculate_efficiency>`
  - ```{autodoc2-docstring} src.pipeline.rl.features.post_processing.calculate_efficiency
    :summary:
    ```
````

### API

`````{py:class} EfficiencyOptimizer(problem, **kwargs)
:canonical: src.pipeline.rl.features.post_processing.EfficiencyOptimizer

```{autodoc2-docstring} src.pipeline.rl.features.post_processing.EfficiencyOptimizer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.rl.features.post_processing.EfficiencyOptimizer.__init__
```

````{py:method} optimize(routes: typing.List[torch.Tensor], **kwargs)
:canonical: src.pipeline.rl.features.post_processing.EfficiencyOptimizer.optimize

```{autodoc2-docstring} src.pipeline.rl.features.post_processing.EfficiencyOptimizer.optimize
```

````

`````

````{py:function} decode_routes(actions: torch.Tensor, num_nodes: int) -> typing.List[typing.List[int]]
:canonical: src.pipeline.rl.features.post_processing.decode_routes

```{autodoc2-docstring} src.pipeline.rl.features.post_processing.decode_routes
```
````

````{py:function} calculate_efficiency(routes, dist_matrix, demand, capacity)
:canonical: src.pipeline.rl.features.post_processing.calculate_efficiency

```{autodoc2-docstring} src.pipeline.rl.features.post_processing.calculate_efficiency
```
````
