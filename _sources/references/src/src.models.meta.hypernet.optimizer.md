# {py:mod}`src.models.meta.hypernet.optimizer`

```{py:module} src.models.meta.hypernet.optimizer
```

```{autodoc2-docstring} src.models.meta.hypernet.optimizer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HyperNetworkOptimizer <src.models.meta.hypernet.optimizer.HyperNetworkOptimizer>`
  - ```{autodoc2-docstring} src.models.meta.hypernet.optimizer.HyperNetworkOptimizer
    :summary:
    ```
````

### API

`````{py:class} HyperNetworkOptimizer(cost_weight_keys: typing.List[str], constraint_value: float, device: torch.device, problem: typing.Any, lr: float = 0.0001, buffer_size: int = 100)
:canonical: src.models.meta.hypernet.optimizer.HyperNetworkOptimizer

```{autodoc2-docstring} src.models.meta.hypernet.optimizer.HyperNetworkOptimizer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.meta.hypernet.optimizer.HyperNetworkOptimizer.__init__
```

````{py:method} update_buffer(metrics: torch.Tensor, day: int, weights: torch.Tensor, performance: float) -> None
:canonical: src.models.meta.hypernet.optimizer.HyperNetworkOptimizer.update_buffer

```{autodoc2-docstring} src.models.meta.hypernet.optimizer.HyperNetworkOptimizer.update_buffer
```

````

````{py:method} train(epochs: int = 10) -> None
:canonical: src.models.meta.hypernet.optimizer.HyperNetworkOptimizer.train

```{autodoc2-docstring} src.models.meta.hypernet.optimizer.HyperNetworkOptimizer.train
```

````

````{py:method} get_weights(all_costs: typing.Dict[str, torch.Tensor], day: int, default_weights: typing.Dict[str, float]) -> typing.Dict[str, float]
:canonical: src.models.meta.hypernet.optimizer.HyperNetworkOptimizer.get_weights

```{autodoc2-docstring} src.models.meta.hypernet.optimizer.HyperNetworkOptimizer.get_weights
```

````

`````
