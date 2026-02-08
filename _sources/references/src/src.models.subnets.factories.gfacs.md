# {py:mod}`src.models.subnets.factories.gfacs`

```{py:module} src.models.subnets.factories.gfacs
```

```{autodoc2-docstring} src.models.subnets.factories.gfacs
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GFACSComponentFactory <src.models.subnets.factories.gfacs.GFACSComponentFactory>`
  - ```{autodoc2-docstring} src.models.subnets.factories.gfacs.GFACSComponentFactory
    :summary:
    ```
````

### API

`````{py:class} GFACSComponentFactory
:canonical: src.models.subnets.factories.gfacs.GFACSComponentFactory

Bases: {py:obj}`src.models.subnets.factories.base.NeuralComponentFactory`

```{autodoc2-docstring} src.models.subnets.factories.gfacs.GFACSComponentFactory
```

````{py:method} create_encoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.gfacs.GFACSComponentFactory.create_encoder

```{autodoc2-docstring} src.models.subnets.factories.gfacs.GFACSComponentFactory.create_encoder
```

````

````{py:method} create_decoder(decoder_type: str = 'aco', **kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.gfacs.GFACSComponentFactory.create_decoder

```{autodoc2-docstring} src.models.subnets.factories.gfacs.GFACSComponentFactory.create_decoder
```

````

`````
