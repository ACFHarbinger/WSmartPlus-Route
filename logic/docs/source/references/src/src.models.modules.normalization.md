# {py:mod}`src.models.modules.normalization`

```{py:module} src.models.modules.normalization
```

```{autodoc2-docstring} src.models.modules.normalization
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Normalization <src.models.modules.normalization.Normalization>`
  - ```{autodoc2-docstring} src.models.modules.normalization.Normalization
    :summary:
    ```
````

### API

`````{py:class} Normalization(embed_dim: int, norm_name: str = 'batch', eps_alpha: float = 1e-05, learn_affine: typing.Optional[bool] = True, track_stats: typing.Optional[bool] = False, mbval: typing.Optional[float] = 0.1, n_groups: typing.Optional[int] = None, kval: typing.Optional[float] = None, bias: typing.Optional[bool] = True)
:canonical: src.models.modules.normalization.Normalization

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.modules.normalization.Normalization
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.modules.normalization.Normalization.__init__
```

````{py:attribute} normalizer
:canonical: src.models.modules.normalization.Normalization.normalizer
:type: torch.nn.Module
:value: >
   None

```{autodoc2-docstring} src.models.modules.normalization.Normalization.normalizer
```

````

````{py:method} init_parameters() -> None
:canonical: src.models.modules.normalization.Normalization.init_parameters

```{autodoc2-docstring} src.models.modules.normalization.Normalization.init_parameters
```

````

````{py:method} forward(input: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.modules.normalization.Normalization.forward

```{autodoc2-docstring} src.models.modules.normalization.Normalization.forward
```

````

`````
