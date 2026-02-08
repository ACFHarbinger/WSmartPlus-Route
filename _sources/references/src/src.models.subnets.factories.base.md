# {py:mod}`src.models.subnets.factories.base`

```{py:module} src.models.subnets.factories.base
```

```{autodoc2-docstring} src.models.subnets.factories.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuralComponentFactory <src.models.subnets.factories.base.NeuralComponentFactory>`
  - ```{autodoc2-docstring} src.models.subnets.factories.base.NeuralComponentFactory
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_create_decoder_by_type <src.models.subnets.factories.base._create_decoder_by_type>`
  - ```{autodoc2-docstring} src.models.subnets.factories.base._create_decoder_by_type
    :summary:
    ```
````

### API

````{py:function} _create_decoder_by_type(decoder_type: str, **kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.base._create_decoder_by_type

```{autodoc2-docstring} src.models.subnets.factories.base._create_decoder_by_type
```
````

`````{py:class} NeuralComponentFactory
:canonical: src.models.subnets.factories.base.NeuralComponentFactory

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.subnets.factories.base.NeuralComponentFactory
```

````{py:method} create_encoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.base.NeuralComponentFactory.create_encoder
:abstractmethod:

```{autodoc2-docstring} src.models.subnets.factories.base.NeuralComponentFactory.create_encoder
```

````

````{py:method} create_decoder(decoder_type: str = 'attention', **kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.base.NeuralComponentFactory.create_decoder
:abstractmethod:

```{autodoc2-docstring} src.models.subnets.factories.base.NeuralComponentFactory.create_decoder
```

````

`````
