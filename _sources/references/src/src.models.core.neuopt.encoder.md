# {py:mod}`src.models.core.neuopt.encoder`

```{py:module} src.models.core.neuopt.encoder
```

```{autodoc2-docstring} src.models.core.neuopt.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeuOptEncoder <src.models.core.neuopt.encoder.NeuOptEncoder>`
  - ```{autodoc2-docstring} src.models.core.neuopt.encoder.NeuOptEncoder
    :summary:
    ```
````

### API

`````{py:class} NeuOptEncoder(embed_dim: int = 128, num_heads: int = 8, num_layers: int = 3, **kwargs: typing.Any)
:canonical: src.models.core.neuopt.encoder.NeuOptEncoder

Bases: {py:obj}`logic.src.models.common.improvement.encoder.ImprovementEncoder`

```{autodoc2-docstring} src.models.core.neuopt.encoder.NeuOptEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.neuopt.encoder.NeuOptEncoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.models.core.neuopt.encoder.NeuOptEncoder.forward

```{autodoc2-docstring} src.models.core.neuopt.encoder.NeuOptEncoder.forward
```

````

`````
