# {py:mod}`src.models.core.dact.encoder`

```{py:module} src.models.core.dact.encoder
```

```{autodoc2-docstring} src.models.core.dact.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DACTEncoder <src.models.core.dact.encoder.DACTEncoder>`
  - ```{autodoc2-docstring} src.models.core.dact.encoder.DACTEncoder
    :summary:
    ```
````

### API

`````{py:class} DACTEncoder(embed_dim: int = 128, num_layers: int = 3, num_heads: int = 8, pos_type: str = 'CPE', **kwargs: typing.Any)
:canonical: src.models.core.dact.encoder.DACTEncoder

Bases: {py:obj}`logic.src.models.common.improvement.encoder.ImprovementEncoder`

```{autodoc2-docstring} src.models.core.dact.encoder.DACTEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.dact.encoder.DACTEncoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.models.core.dact.encoder.DACTEncoder.forward

```{autodoc2-docstring} src.models.core.dact.encoder.DACTEncoder.forward
```

````

`````
