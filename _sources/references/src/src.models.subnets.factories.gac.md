# {py:mod}`src.models.subnets.factories.gac`

```{py:module} src.models.subnets.factories.gac
```

```{autodoc2-docstring} src.models.subnets.factories.gac
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GACComponentFactory <src.models.subnets.factories.gac.GACComponentFactory>`
  - ```{autodoc2-docstring} src.models.subnets.factories.gac.GACComponentFactory
    :summary:
    ```
````

### API

`````{py:class} GACComponentFactory
:canonical: src.models.subnets.factories.gac.GACComponentFactory

Bases: {py:obj}`src.models.subnets.factories.base.NeuralComponentFactory`

```{autodoc2-docstring} src.models.subnets.factories.gac.GACComponentFactory
```

````{py:method} create_encoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.gac.GACComponentFactory.create_encoder

```{autodoc2-docstring} src.models.subnets.factories.gac.GACComponentFactory.create_encoder
```

````

````{py:method} create_decoder(decoder_type: str = 'attention', **kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.gac.GACComponentFactory.create_decoder

```{autodoc2-docstring} src.models.subnets.factories.gac.GACComponentFactory.create_decoder
```

````

`````
