# {py:mod}`src.models.common.autoregressive_encoder`

```{py:module} src.models.common.autoregressive_encoder
```

```{autodoc2-docstring} src.models.common.autoregressive_encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AutoregressiveEncoder <src.models.common.autoregressive_encoder.AutoregressiveEncoder>`
  - ```{autodoc2-docstring} src.models.common.autoregressive_encoder.AutoregressiveEncoder
    :summary:
    ```
````

### API

`````{py:class} AutoregressiveEncoder(embed_dim: int = 128, **kwargs)
:canonical: src.models.common.autoregressive_encoder.AutoregressiveEncoder

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.common.autoregressive_encoder.AutoregressiveEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.autoregressive_encoder.AutoregressiveEncoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, **kwargs) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]]
:canonical: src.models.common.autoregressive_encoder.AutoregressiveEncoder.forward
:abstractmethod:

```{autodoc2-docstring} src.models.common.autoregressive_encoder.AutoregressiveEncoder.forward
```

````

`````
