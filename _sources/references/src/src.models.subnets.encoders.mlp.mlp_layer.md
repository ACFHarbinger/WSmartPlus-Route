# {py:mod}`src.models.subnets.encoders.mlp.mlp_layer`

```{py:module} src.models.subnets.encoders.mlp.mlp_layer
```

```{autodoc2-docstring} src.models.subnets.encoders.mlp.mlp_layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MLPLayer <src.models.subnets.encoders.mlp.mlp_layer.MLPLayer>`
  - ```{autodoc2-docstring} src.models.subnets.encoders.mlp.mlp_layer.MLPLayer
    :summary:
    ```
````

### API

`````{py:class} MLPLayer(hidden_dim: int, norm_config: logic.src.configs.models.normalization.NormalizationConfig, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None)
:canonical: src.models.subnets.encoders.mlp.mlp_layer.MLPLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.encoders.mlp.mlp_layer.MLPLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.encoders.mlp.mlp_layer.MLPLayer.__init__
```

````{py:method} forward(x)
:canonical: src.models.subnets.encoders.mlp.mlp_layer.MLPLayer.forward

```{autodoc2-docstring} src.models.subnets.encoders.mlp.mlp_layer.MLPLayer.forward
```

````

`````
