# {py:mod}`src.models.subnets.modules.moe_layer`

```{py:module} src.models.subnets.modules.moe_layer
```

```{autodoc2-docstring} src.models.subnets.modules.moe_layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MoE <src.models.subnets.modules.moe_layer.MoE>`
  - ```{autodoc2-docstring} src.models.subnets.modules.moe_layer.MoE
    :summary:
    ```
````

### API

`````{py:class} MoE(input_size, output_size, num_neurons=None, experts=None, hidden_act='ReLU', out_bias=True, num_experts=4, k=2, noisy_gating=True, **kwargs)
:canonical: src.models.subnets.modules.moe_layer.MoE

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.modules.moe_layer.MoE
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.moe_layer.MoE.__init__
```

````{py:method} cv_squared(x)
:canonical: src.models.subnets.modules.moe_layer.MoE.cv_squared

```{autodoc2-docstring} src.models.subnets.modules.moe_layer.MoE.cv_squared
```

````

````{py:method} _gates_to_load(gates)
:canonical: src.models.subnets.modules.moe_layer.MoE._gates_to_load

```{autodoc2-docstring} src.models.subnets.modules.moe_layer.MoE._gates_to_load
```

````

````{py:method} _prob_in_top_k(clean_values, noisy_values, noise_stddev, noisy_top_values)
:canonical: src.models.subnets.modules.moe_layer.MoE._prob_in_top_k

```{autodoc2-docstring} src.models.subnets.modules.moe_layer.MoE._prob_in_top_k
```

````

````{py:method} noisy_top_k_gating(x, train, noise_epsilon=0.01)
:canonical: src.models.subnets.modules.moe_layer.MoE.noisy_top_k_gating

```{autodoc2-docstring} src.models.subnets.modules.moe_layer.MoE.noisy_top_k_gating
```

````

````{py:method} forward(x, loss_coef=0.0)
:canonical: src.models.subnets.modules.moe_layer.MoE.forward

```{autodoc2-docstring} src.models.subnets.modules.moe_layer.MoE.forward
```

````

`````
