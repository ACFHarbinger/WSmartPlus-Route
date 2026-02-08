# {py:mod}`src.models.common.improvement_encoder`

```{py:module} src.models.common.improvement_encoder
```

```{autodoc2-docstring} src.models.common.improvement_encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImprovementEncoder <src.models.common.improvement_encoder.ImprovementEncoder>`
  - ```{autodoc2-docstring} src.models.common.improvement_encoder.ImprovementEncoder
    :summary:
    ```
````

### API

`````{py:class} ImprovementEncoder(embed_dim: int = 128, **kwargs)
:canonical: src.models.common.improvement_encoder.ImprovementEncoder

Bases: {py:obj}`torch.nn.Module`, {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.common.improvement_encoder.ImprovementEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.improvement_encoder.ImprovementEncoder.__init__
```

````{py:method} forward(td: tensordict.TensorDict, **kwargs) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]]
:canonical: src.models.common.improvement_encoder.ImprovementEncoder.forward
:abstractmethod:

```{autodoc2-docstring} src.models.common.improvement_encoder.ImprovementEncoder.forward
```

````

`````
