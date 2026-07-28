# {py:mod}`src.models.subnets.factories.mlp`

```{py:module} src.models.subnets.factories.mlp
```

```{autodoc2-docstring} src.models.subnets.factories.mlp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MLPComponentFactory <src.models.subnets.factories.mlp.MLPComponentFactory>`
  - ```{autodoc2-docstring} src.models.subnets.factories.mlp.MLPComponentFactory
    :summary:
    ```
````

### API

`````{py:class} MLPComponentFactory
:canonical: src.models.subnets.factories.mlp.MLPComponentFactory

Bases: {py:obj}`src.models.subnets.factories.base.NeuralComponentFactory`

```{autodoc2-docstring} src.models.subnets.factories.mlp.MLPComponentFactory
```

````{py:method} create_encoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.mlp.MLPComponentFactory.create_encoder

```{autodoc2-docstring} src.models.subnets.factories.mlp.MLPComponentFactory.create_encoder
```

````

````{py:method} create_decoder(decoder_type: str = 'attention', **kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.mlp.MLPComponentFactory.create_decoder

```{autodoc2-docstring} src.models.subnets.factories.mlp.MLPComponentFactory.create_decoder
```

````

`````
