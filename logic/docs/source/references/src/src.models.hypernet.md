# {py:mod}`src.models.hypernet`

```{py:module} src.models.hypernet
```

```{autodoc2-docstring} src.models.hypernet
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Hypernetwork <src.models.hypernet.Hypernetwork>`
  - ```{autodoc2-docstring} src.models.hypernet.Hypernetwork
    :summary:
    ```
* - {py:obj}`HypernetworkOptimizer <src.models.hypernet.HypernetworkOptimizer>`
  - ```{autodoc2-docstring} src.models.hypernet.HypernetworkOptimizer
    :summary:
    ```
````

### API

`````{py:class} Hypernetwork(problem, n_days=365, embed_dim=16, hidden_dim=64, normalization='layer', activation='relu', learn_affine=True, bias=True)
:canonical: src.models.hypernet.Hypernetwork

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.hypernet.Hypernetwork
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.hypernet.Hypernetwork.__init__
```

````{py:method} init_weights()
:canonical: src.models.hypernet.Hypernetwork.init_weights

```{autodoc2-docstring} src.models.hypernet.Hypernetwork.init_weights
```

````

````{py:method} forward(metrics, day)
:canonical: src.models.hypernet.Hypernetwork.forward

```{autodoc2-docstring} src.models.hypernet.Hypernetwork.forward
```

````

`````

`````{py:class} HypernetworkOptimizer(cost_weight_keys, constraint_value, device, problem, lr=0.0001, buffer_size=100)
:canonical: src.models.hypernet.HypernetworkOptimizer

```{autodoc2-docstring} src.models.hypernet.HypernetworkOptimizer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.hypernet.HypernetworkOptimizer.__init__
```

````{py:method} update_buffer(metrics, day, weights, performance)
:canonical: src.models.hypernet.HypernetworkOptimizer.update_buffer

```{autodoc2-docstring} src.models.hypernet.HypernetworkOptimizer.update_buffer
```

````

````{py:method} train(epochs=10)
:canonical: src.models.hypernet.HypernetworkOptimizer.train

```{autodoc2-docstring} src.models.hypernet.HypernetworkOptimizer.train
```

````

````{py:method} get_weights(all_costs, day, default_weights)
:canonical: src.models.hypernet.HypernetworkOptimizer.get_weights

```{autodoc2-docstring} src.models.hypernet.HypernetworkOptimizer.get_weights
```

````

`````
