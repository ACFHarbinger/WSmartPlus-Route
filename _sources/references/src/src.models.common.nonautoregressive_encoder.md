# {py:mod}`src.models.common.nonautoregressive_encoder`

```{py:module} src.models.common.nonautoregressive_encoder
```

```{autodoc2-docstring} src.models.common.nonautoregressive_encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NonAutoregressiveEncoder <src.models.common.nonautoregressive_encoder.NonAutoregressiveEncoder>`
  - ```{autodoc2-docstring} src.models.common.nonautoregressive_encoder.NonAutoregressiveEncoder
    :summary:
    ```
````

### API

`````{py:class} NonAutoregressiveEncoder(embed_dim: int = 128, **kwargs)
:canonical: src.models.common.nonautoregressive_encoder.NonAutoregressiveEncoder

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.common.nonautoregressive_encoder.NonAutoregressiveEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.nonautoregressive_encoder.NonAutoregressiveEncoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, **kwargs) -> torch.Tensor | tuple[torch.Tensor, ...]
:canonical: src.models.common.nonautoregressive_encoder.NonAutoregressiveEncoder.forward
:abstractmethod:

```{autodoc2-docstring} src.models.common.nonautoregressive_encoder.NonAutoregressiveEncoder.forward
```

````

`````
