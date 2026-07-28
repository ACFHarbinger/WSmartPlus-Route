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

`````{py:class} MoE(input_size: int, output_size: int, num_neurons: typing.Optional[list] = None, experts: typing.Optional[torch.nn.ModuleList | list] = None, hidden_act: str = 'ReLU', out_bias: bool = True, num_experts: int = 4, k: int = 2, noisy_gating: bool = True, **kwargs: typing.Any)
:canonical: src.models.subnets.modules.moe_layer.MoE

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.modules.moe_layer.MoE
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.moe_layer.MoE.__init__
```

````{py:method} cv_squared(x: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.modules.moe_layer.MoE.cv_squared

```{autodoc2-docstring} src.models.subnets.modules.moe_layer.MoE.cv_squared
```

````

````{py:method} _gates_to_load(gates: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.modules.moe_layer.MoE._gates_to_load

```{autodoc2-docstring} src.models.subnets.modules.moe_layer.MoE._gates_to_load
```

````

````{py:method} _prob_in_top_k(clean_values: torch.Tensor, noisy_values: torch.Tensor, noise_stddev: torch.Tensor, noisy_top_values: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.modules.moe_layer.MoE._prob_in_top_k

```{autodoc2-docstring} src.models.subnets.modules.moe_layer.MoE._prob_in_top_k
```

````

````{py:method} noisy_top_k_gating(x: torch.Tensor, train: bool, noise_epsilon: float = 0.01) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.subnets.modules.moe_layer.MoE.noisy_top_k_gating

```{autodoc2-docstring} src.models.subnets.modules.moe_layer.MoE.noisy_top_k_gating
```

````

````{py:method} forward(x: torch.Tensor, loss_coef: float = 0.0) -> torch.Tensor
:canonical: src.models.subnets.modules.moe_layer.MoE.forward

```{autodoc2-docstring} src.models.subnets.modules.moe_layer.MoE.forward
```

````

`````
