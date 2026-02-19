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

`````{py:class} HyperNetworkOptimizer(cost_weight_keys, constraint_value, device, problem, lr=0.0001, buffer_size=100)
:canonical: src.models.meta.hypernet.optimizer.HyperNetworkOptimizer

```{autodoc2-docstring} src.models.meta.hypernet.optimizer.HyperNetworkOptimizer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.meta.hypernet.optimizer.HyperNetworkOptimizer.__init__
```

````{py:method} update_buffer(metrics, day, weights, performance)
:canonical: src.models.meta.hypernet.optimizer.HyperNetworkOptimizer.update_buffer

```{autodoc2-docstring} src.models.meta.hypernet.optimizer.HyperNetworkOptimizer.update_buffer
```

````

````{py:method} train(epochs=10)
:canonical: src.models.meta.hypernet.optimizer.HyperNetworkOptimizer.train

```{autodoc2-docstring} src.models.meta.hypernet.optimizer.HyperNetworkOptimizer.train
```

````

````{py:method} get_weights(all_costs, day, default_weights)
:canonical: src.models.meta.hypernet.optimizer.HyperNetworkOptimizer.get_weights

```{autodoc2-docstring} src.models.meta.hypernet.optimizer.HyperNetworkOptimizer.get_weights
```

````

`````
