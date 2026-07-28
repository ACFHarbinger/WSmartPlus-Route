# {py:mod}`src.models.subnets.factories.gcn`

```{py:module} src.models.subnets.factories.gcn
```

```{autodoc2-docstring} src.models.subnets.factories.gcn
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GCNComponentFactory <src.models.subnets.factories.gcn.GCNComponentFactory>`
  - ```{autodoc2-docstring} src.models.subnets.factories.gcn.GCNComponentFactory
    :summary:
    ```
````

### API

`````{py:class} GCNComponentFactory
:canonical: src.models.subnets.factories.gcn.GCNComponentFactory

Bases: {py:obj}`src.models.subnets.factories.base.NeuralComponentFactory`

```{autodoc2-docstring} src.models.subnets.factories.gcn.GCNComponentFactory
```

````{py:method} create_encoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.gcn.GCNComponentFactory.create_encoder

```{autodoc2-docstring} src.models.subnets.factories.gcn.GCNComponentFactory.create_encoder
```

````

````{py:method} create_decoder(decoder_type: str = 'attention', **kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.gcn.GCNComponentFactory.create_decoder

```{autodoc2-docstring} src.models.subnets.factories.gcn.GCNComponentFactory.create_decoder
```

````

`````
