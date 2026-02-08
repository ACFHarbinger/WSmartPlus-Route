# {py:mod}`src.models.subnets.factories.mdam`

```{py:module} src.models.subnets.factories.mdam
```

```{autodoc2-docstring} src.models.subnets.factories.mdam
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MDAMComponentFactory <src.models.subnets.factories.mdam.MDAMComponentFactory>`
  - ```{autodoc2-docstring} src.models.subnets.factories.mdam.MDAMComponentFactory
    :summary:
    ```
````

### API

`````{py:class} MDAMComponentFactory
:canonical: src.models.subnets.factories.mdam.MDAMComponentFactory

Bases: {py:obj}`src.models.subnets.factories.base.NeuralComponentFactory`

```{autodoc2-docstring} src.models.subnets.factories.mdam.MDAMComponentFactory
```

````{py:method} create_encoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.mdam.MDAMComponentFactory.create_encoder

```{autodoc2-docstring} src.models.subnets.factories.mdam.MDAMComponentFactory.create_encoder
```

````

````{py:method} create_decoder(decoder_type: str = 'mdam', **kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.mdam.MDAMComponentFactory.create_decoder

```{autodoc2-docstring} src.models.subnets.factories.mdam.MDAMComponentFactory.create_decoder
```

````

`````
