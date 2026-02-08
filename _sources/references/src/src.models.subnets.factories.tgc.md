# {py:mod}`src.models.subnets.factories.tgc`

```{py:module} src.models.subnets.factories.tgc
```

```{autodoc2-docstring} src.models.subnets.factories.tgc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TGCComponentFactory <src.models.subnets.factories.tgc.TGCComponentFactory>`
  - ```{autodoc2-docstring} src.models.subnets.factories.tgc.TGCComponentFactory
    :summary:
    ```
````

### API

`````{py:class} TGCComponentFactory
:canonical: src.models.subnets.factories.tgc.TGCComponentFactory

Bases: {py:obj}`src.models.subnets.factories.base.NeuralComponentFactory`

```{autodoc2-docstring} src.models.subnets.factories.tgc.TGCComponentFactory
```

````{py:method} create_encoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.tgc.TGCComponentFactory.create_encoder

```{autodoc2-docstring} src.models.subnets.factories.tgc.TGCComponentFactory.create_encoder
```

````

````{py:method} create_decoder(decoder_type: str = 'attention', **kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.tgc.TGCComponentFactory.create_decoder

```{autodoc2-docstring} src.models.subnets.factories.tgc.TGCComponentFactory.create_decoder
```

````

`````
