# {py:mod}`src.models.subnets.factories.attention`

```{py:module} src.models.subnets.factories.attention
```

```{autodoc2-docstring} src.models.subnets.factories.attention
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AttentionComponentFactory <src.models.subnets.factories.attention.AttentionComponentFactory>`
  - ```{autodoc2-docstring} src.models.subnets.factories.attention.AttentionComponentFactory
    :summary:
    ```
````

### API

`````{py:class} AttentionComponentFactory
:canonical: src.models.subnets.factories.attention.AttentionComponentFactory

Bases: {py:obj}`src.models.subnets.factories.base.NeuralComponentFactory`

```{autodoc2-docstring} src.models.subnets.factories.attention.AttentionComponentFactory
```

````{py:method} create_encoder(norm_config: typing.Optional[logic.src.configs.models.normalization.NormalizationConfig] = None, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None, **kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.attention.AttentionComponentFactory.create_encoder

```{autodoc2-docstring} src.models.subnets.factories.attention.AttentionComponentFactory.create_encoder
```

````

````{py:method} create_decoder(decoder_type: str = 'attention', norm_config: typing.Optional[logic.src.configs.models.normalization.NormalizationConfig] = None, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None, **kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.attention.AttentionComponentFactory.create_decoder

```{autodoc2-docstring} src.models.subnets.factories.attention.AttentionComponentFactory.create_decoder
```

````

`````
