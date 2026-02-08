# {py:mod}`src.models.dact.encoder`

```{py:module} src.models.dact.encoder
```

```{autodoc2-docstring} src.models.dact.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DACTEncoder <src.models.dact.encoder.DACTEncoder>`
  - ```{autodoc2-docstring} src.models.dact.encoder.DACTEncoder
    :summary:
    ```
````

### API

`````{py:class} DACTEncoder(embed_dim: int = 128, num_layers: int = 3, num_heads: int = 8, pos_type: str = 'CPE', **kwargs)
:canonical: src.models.dact.encoder.DACTEncoder

Bases: {py:obj}`src.models.common.improvement_encoder.ImprovementEncoder`

```{autodoc2-docstring} src.models.dact.encoder.DACTEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.dact.encoder.DACTEncoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, **kwargs) -> torch.Tensor
:canonical: src.models.dact.encoder.DACTEncoder.forward

```{autodoc2-docstring} src.models.dact.encoder.DACTEncoder.forward
```

````

`````
