# {py:mod}`src.models.subnets.factories.nar`

```{py:module} src.models.subnets.factories.nar
```

```{autodoc2-docstring} src.models.subnets.factories.nar
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NARComponentFactory <src.models.subnets.factories.nar.NARComponentFactory>`
  - ```{autodoc2-docstring} src.models.subnets.factories.nar.NARComponentFactory
    :summary:
    ```
````

### API

`````{py:class} NARComponentFactory
:canonical: src.models.subnets.factories.nar.NARComponentFactory

Bases: {py:obj}`src.models.subnets.factories.base.NeuralComponentFactory`

```{autodoc2-docstring} src.models.subnets.factories.nar.NARComponentFactory
```

````{py:method} create_encoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.nar.NARComponentFactory.create_encoder

```{autodoc2-docstring} src.models.subnets.factories.nar.NARComponentFactory.create_encoder
```

````

````{py:method} create_decoder(decoder_type: str = 'aco', **kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.nar.NARComponentFactory.create_decoder

```{autodoc2-docstring} src.models.subnets.factories.nar.NARComponentFactory.create_decoder
```

````

`````
