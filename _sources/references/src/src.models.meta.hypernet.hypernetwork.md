# {py:mod}`src.models.meta.hypernet.hypernetwork`

```{py:module} src.models.meta.hypernet.hypernetwork
```

```{autodoc2-docstring} src.models.meta.hypernet.hypernetwork
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HyperNetwork <src.models.meta.hypernet.hypernetwork.HyperNetwork>`
  - ```{autodoc2-docstring} src.models.meta.hypernet.hypernetwork.HyperNetwork
    :summary:
    ```
````

### API

`````{py:class} HyperNetwork(input_dim: int, output_dim: int, n_days: int = 365, embed_dim: int = 16, hidden_dim: int = 64, normalization: str = 'layer', activation: str = 'relu', learn_affine: bool = True, bias: bool = True)
:canonical: src.models.meta.hypernet.hypernetwork.HyperNetwork

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.meta.hypernet.hypernetwork.HyperNetwork
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.meta.hypernet.hypernetwork.HyperNetwork.__init__
```

````{py:method} init_weights() -> None
:canonical: src.models.meta.hypernet.hypernetwork.HyperNetwork.init_weights

```{autodoc2-docstring} src.models.meta.hypernet.hypernetwork.HyperNetwork.init_weights
```

````

````{py:method} forward(metrics: torch.Tensor, day: torch.Tensor) -> torch.Tensor
:canonical: src.models.meta.hypernet.hypernetwork.HyperNetwork.forward

```{autodoc2-docstring} src.models.meta.hypernet.hypernetwork.HyperNetwork.forward
```

````

`````
