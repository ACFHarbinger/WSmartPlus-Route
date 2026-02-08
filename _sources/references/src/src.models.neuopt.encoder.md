# {py:mod}`src.models.neuopt.encoder`

```{py:module} src.models.neuopt.encoder
```

```{autodoc2-docstring} src.models.neuopt.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuOptEncoder <src.models.neuopt.encoder.NeuOptEncoder>`
  - ```{autodoc2-docstring} src.models.neuopt.encoder.NeuOptEncoder
    :summary:
    ```
````

### API

`````{py:class} NeuOptEncoder(embed_dim: int = 128, num_heads: int = 8, num_layers: int = 3, **kwargs)
:canonical: src.models.neuopt.encoder.NeuOptEncoder

Bases: {py:obj}`src.models.common.improvement_encoder.ImprovementEncoder`

```{autodoc2-docstring} src.models.neuopt.encoder.NeuOptEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.neuopt.encoder.NeuOptEncoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, **kwargs) -> torch.Tensor
:canonical: src.models.neuopt.encoder.NeuOptEncoder.forward

```{autodoc2-docstring} src.models.neuopt.encoder.NeuOptEncoder.forward
```

````

`````
