# {py:mod}`src.models.subnets.factories.moe`

```{py:module} src.models.subnets.factories.moe
```

```{autodoc2-docstring} src.models.subnets.factories.moe
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MoEComponentFactory <src.models.subnets.factories.moe.MoEComponentFactory>`
  - ```{autodoc2-docstring} src.models.subnets.factories.moe.MoEComponentFactory
    :summary:
    ```
````

### API

`````{py:class} MoEComponentFactory(num_experts: int = 4, k: int = 2, noisy_gating: bool = True)
:canonical: src.models.subnets.factories.moe.MoEComponentFactory

Bases: {py:obj}`src.models.subnets.factories.base.NeuralComponentFactory`

```{autodoc2-docstring} src.models.subnets.factories.moe.MoEComponentFactory
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.factories.moe.MoEComponentFactory.__init__
```

````{py:method} create_encoder(**kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.moe.MoEComponentFactory.create_encoder

```{autodoc2-docstring} src.models.subnets.factories.moe.MoEComponentFactory.create_encoder
```

````

````{py:method} create_decoder(decoder_type: str = 'attention', **kwargs: typing.Any) -> torch.nn.Module
:canonical: src.models.subnets.factories.moe.MoEComponentFactory.create_decoder

```{autodoc2-docstring} src.models.subnets.factories.moe.MoEComponentFactory.create_decoder
```

````

`````
