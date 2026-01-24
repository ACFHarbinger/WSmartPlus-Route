# {py:mod}`src.models.model_factory`

```{py:module} src.models.model_factory
```

```{autodoc2-docstring} src.models.model_factory
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuralComponentFactory <src.models.model_factory.NeuralComponentFactory>`
  - ```{autodoc2-docstring} src.models.model_factory.NeuralComponentFactory
    :summary:
    ```
* - {py:obj}`AttentionComponentFactory <src.models.model_factory.AttentionComponentFactory>`
  - ```{autodoc2-docstring} src.models.model_factory.AttentionComponentFactory
    :summary:
    ```
* - {py:obj}`GCNComponentFactory <src.models.model_factory.GCNComponentFactory>`
  - ```{autodoc2-docstring} src.models.model_factory.GCNComponentFactory
    :summary:
    ```
* - {py:obj}`GACComponentFactory <src.models.model_factory.GACComponentFactory>`
  - ```{autodoc2-docstring} src.models.model_factory.GACComponentFactory
    :summary:
    ```
* - {py:obj}`TGCComponentFactory <src.models.model_factory.TGCComponentFactory>`
  - ```{autodoc2-docstring} src.models.model_factory.TGCComponentFactory
    :summary:
    ```
* - {py:obj}`GGACComponentFactory <src.models.model_factory.GGACComponentFactory>`
  - ```{autodoc2-docstring} src.models.model_factory.GGACComponentFactory
    :summary:
    ```
* - {py:obj}`MLPComponentFactory <src.models.model_factory.MLPComponentFactory>`
  - ```{autodoc2-docstring} src.models.model_factory.MLPComponentFactory
    :summary:
    ```
* - {py:obj}`MoEComponentFactory <src.models.model_factory.MoEComponentFactory>`
  - ```{autodoc2-docstring} src.models.model_factory.MoEComponentFactory
    :summary:
    ```
````

### API

`````{py:class} NeuralComponentFactory
:canonical: src.models.model_factory.NeuralComponentFactory

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.model_factory.NeuralComponentFactory
```

````{py:method} create_encoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.model_factory.NeuralComponentFactory.create_encoder
:abstractmethod:

```{autodoc2-docstring} src.models.model_factory.NeuralComponentFactory.create_encoder
```

````

````{py:method} create_decoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.model_factory.NeuralComponentFactory.create_decoder
:abstractmethod:

```{autodoc2-docstring} src.models.model_factory.NeuralComponentFactory.create_decoder
```

````

`````

`````{py:class} AttentionComponentFactory
:canonical: src.models.model_factory.AttentionComponentFactory

Bases: {py:obj}`src.models.model_factory.NeuralComponentFactory`

```{autodoc2-docstring} src.models.model_factory.AttentionComponentFactory
```

````{py:method} create_encoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.model_factory.AttentionComponentFactory.create_encoder

```{autodoc2-docstring} src.models.model_factory.AttentionComponentFactory.create_encoder
```

````

````{py:method} create_decoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.model_factory.AttentionComponentFactory.create_decoder

```{autodoc2-docstring} src.models.model_factory.AttentionComponentFactory.create_decoder
```

````

`````

`````{py:class} GCNComponentFactory
:canonical: src.models.model_factory.GCNComponentFactory

Bases: {py:obj}`src.models.model_factory.NeuralComponentFactory`

```{autodoc2-docstring} src.models.model_factory.GCNComponentFactory
```

````{py:method} create_encoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.model_factory.GCNComponentFactory.create_encoder

```{autodoc2-docstring} src.models.model_factory.GCNComponentFactory.create_encoder
```

````

````{py:method} create_decoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.model_factory.GCNComponentFactory.create_decoder

```{autodoc2-docstring} src.models.model_factory.GCNComponentFactory.create_decoder
```

````

`````

`````{py:class} GACComponentFactory
:canonical: src.models.model_factory.GACComponentFactory

Bases: {py:obj}`src.models.model_factory.NeuralComponentFactory`

```{autodoc2-docstring} src.models.model_factory.GACComponentFactory
```

````{py:method} create_encoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.model_factory.GACComponentFactory.create_encoder

```{autodoc2-docstring} src.models.model_factory.GACComponentFactory.create_encoder
```

````

````{py:method} create_decoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.model_factory.GACComponentFactory.create_decoder

```{autodoc2-docstring} src.models.model_factory.GACComponentFactory.create_decoder
```

````

`````

`````{py:class} TGCComponentFactory
:canonical: src.models.model_factory.TGCComponentFactory

Bases: {py:obj}`src.models.model_factory.NeuralComponentFactory`

```{autodoc2-docstring} src.models.model_factory.TGCComponentFactory
```

````{py:method} create_encoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.model_factory.TGCComponentFactory.create_encoder

```{autodoc2-docstring} src.models.model_factory.TGCComponentFactory.create_encoder
```

````

````{py:method} create_decoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.model_factory.TGCComponentFactory.create_decoder

```{autodoc2-docstring} src.models.model_factory.TGCComponentFactory.create_decoder
```

````

`````

`````{py:class} GGACComponentFactory
:canonical: src.models.model_factory.GGACComponentFactory

Bases: {py:obj}`src.models.model_factory.NeuralComponentFactory`

```{autodoc2-docstring} src.models.model_factory.GGACComponentFactory
```

````{py:method} create_encoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.model_factory.GGACComponentFactory.create_encoder

```{autodoc2-docstring} src.models.model_factory.GGACComponentFactory.create_encoder
```

````

````{py:method} create_decoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.model_factory.GGACComponentFactory.create_decoder

```{autodoc2-docstring} src.models.model_factory.GGACComponentFactory.create_decoder
```

````

`````

`````{py:class} MLPComponentFactory
:canonical: src.models.model_factory.MLPComponentFactory

Bases: {py:obj}`src.models.model_factory.NeuralComponentFactory`

```{autodoc2-docstring} src.models.model_factory.MLPComponentFactory
```

````{py:method} create_encoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.model_factory.MLPComponentFactory.create_encoder

```{autodoc2-docstring} src.models.model_factory.MLPComponentFactory.create_encoder
```

````

````{py:method} create_decoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.model_factory.MLPComponentFactory.create_decoder

```{autodoc2-docstring} src.models.model_factory.MLPComponentFactory.create_decoder
```

````

`````

`````{py:class} MoEComponentFactory(num_experts: int = 4, k: int = 2, noisy_gating: bool = True)
:canonical: src.models.model_factory.MoEComponentFactory

Bases: {py:obj}`src.models.model_factory.NeuralComponentFactory`

```{autodoc2-docstring} src.models.model_factory.MoEComponentFactory
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.model_factory.MoEComponentFactory.__init__
```

````{py:method} create_encoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.model_factory.MoEComponentFactory.create_encoder

```{autodoc2-docstring} src.models.model_factory.MoEComponentFactory.create_encoder
```

````

````{py:method} create_decoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.model_factory.MoEComponentFactory.create_decoder

```{autodoc2-docstring} src.models.model_factory.MoEComponentFactory.create_decoder
```

````

`````
