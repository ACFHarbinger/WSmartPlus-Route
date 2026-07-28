# {py:mod}`src.models.subnets.factories.ggac`

```{py:module} src.models.subnets.factories.ggac
```

```{autodoc2-docstring} src.models.subnets.factories.ggac
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GGACComponentFactory <src.models.subnets.factories.ggac.GGACComponentFactory>`
  - ```{autodoc2-docstring} src.models.subnets.factories.ggac.GGACComponentFactory
    :summary:
    ```
````

### API

`````{py:class} GGACComponentFactory
:canonical: src.models.subnets.factories.ggac.GGACComponentFactory

Bases: {py:obj}`src.models.subnets.factories.base.NeuralComponentFactory`

```{autodoc2-docstring} src.models.subnets.factories.ggac.GGACComponentFactory
```

````{py:method} create_encoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.ggac.GGACComponentFactory.create_encoder

```{autodoc2-docstring} src.models.subnets.factories.ggac.GGACComponentFactory.create_encoder
```

````

````{py:method} create_decoder(decoder_type: str = 'attention', **kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.ggac.GGACComponentFactory.create_decoder

```{autodoc2-docstring} src.models.subnets.factories.ggac.GGACComponentFactory.create_decoder
```

````

`````
