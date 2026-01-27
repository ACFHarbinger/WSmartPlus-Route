# {py:mod}`src.models.modules.moe`

```{py:module} src.models.modules.moe
```

```{autodoc2-docstring} src.models.modules.moe
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SparseDispatcher <src.models.modules.moe.SparseDispatcher>`
  - ```{autodoc2-docstring} src.models.modules.moe.SparseDispatcher
    :summary:
    ```
* - {py:obj}`MoE <src.models.modules.moe.MoE>`
  - ```{autodoc2-docstring} src.models.modules.moe.MoE
    :summary:
    ```
````

### API

`````{py:class} SparseDispatcher(num_experts, gates)
:canonical: src.models.modules.moe.SparseDispatcher

Bases: {py:obj}`object`

```{autodoc2-docstring} src.models.modules.moe.SparseDispatcher
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.modules.moe.SparseDispatcher.__init__
```

````{py:method} dispatch(inp)
:canonical: src.models.modules.moe.SparseDispatcher.dispatch

```{autodoc2-docstring} src.models.modules.moe.SparseDispatcher.dispatch
```

````

````{py:method} combine(expert_out, multiply_by_gates=True)
:canonical: src.models.modules.moe.SparseDispatcher.combine

```{autodoc2-docstring} src.models.modules.moe.SparseDispatcher.combine
```

````

````{py:method} expert_to_gates()
:canonical: src.models.modules.moe.SparseDispatcher.expert_to_gates

```{autodoc2-docstring} src.models.modules.moe.SparseDispatcher.expert_to_gates
```

````

`````

`````{py:class} MoE(input_size, output_size, num_neurons=None, experts=None, hidden_act='ReLU', out_bias=True, num_experts=4, k=2, noisy_gating=True, **kwargs)
:canonical: src.models.modules.moe.MoE

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.modules.moe.MoE
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.modules.moe.MoE.__init__
```

````{py:method} cv_squared(x)
:canonical: src.models.modules.moe.MoE.cv_squared

```{autodoc2-docstring} src.models.modules.moe.MoE.cv_squared
```

````

````{py:method} _gates_to_load(gates)
:canonical: src.models.modules.moe.MoE._gates_to_load

```{autodoc2-docstring} src.models.modules.moe.MoE._gates_to_load
```

````

````{py:method} _prob_in_top_k(clean_values, noisy_values, noise_stddev, noisy_top_values)
:canonical: src.models.modules.moe.MoE._prob_in_top_k

```{autodoc2-docstring} src.models.modules.moe.MoE._prob_in_top_k
```

````

````{py:method} noisy_top_k_gating(x, train, noise_epsilon=0.01)
:canonical: src.models.modules.moe.MoE.noisy_top_k_gating

```{autodoc2-docstring} src.models.modules.moe.MoE.noisy_top_k_gating
```

````

````{py:method} forward(x, loss_coef=0.0)
:canonical: src.models.modules.moe.MoE.forward

```{autodoc2-docstring} src.models.modules.moe.MoE.forward
```

````

`````
