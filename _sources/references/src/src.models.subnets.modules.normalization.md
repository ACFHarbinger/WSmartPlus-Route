# {py:mod}`src.models.subnets.modules.normalization`

```{py:module} src.models.subnets.modules.normalization
```

```{autodoc2-docstring} src.models.subnets.modules.normalization
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Normalization <src.models.subnets.modules.normalization.Normalization>`
  - ```{autodoc2-docstring} src.models.subnets.modules.normalization.Normalization
    :summary:
    ```
````

### API

`````{py:class} Normalization(embed_dim: int, norm_name: typing.Optional[str] = None, eps_alpha: typing.Optional[float] = None, learn_affine: typing.Optional[bool] = None, track_stats: typing.Optional[bool] = None, mbval: typing.Optional[float] = None, n_groups: typing.Optional[int] = None, kval: typing.Optional[float] = None, bias: typing.Optional[bool] = True, norm_config: typing.Optional[logic.src.configs.models.normalization.NormalizationConfig] = None)
:canonical: src.models.subnets.modules.normalization.Normalization

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.modules.normalization.Normalization
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.normalization.Normalization.__init__
```

````{py:attribute} normalizer
:canonical: src.models.subnets.modules.normalization.Normalization.normalizer
:type: torch.nn.Module
:value: >
   None

```{autodoc2-docstring} src.models.subnets.modules.normalization.Normalization.normalizer
```

````

````{py:method} init_parameters() -> None
:canonical: src.models.subnets.modules.normalization.Normalization.init_parameters

```{autodoc2-docstring} src.models.subnets.modules.normalization.Normalization.init_parameters
```

````

````{py:method} forward(input: torch.Tensor, mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor
:canonical: src.models.subnets.modules.normalization.Normalization.forward

```{autodoc2-docstring} src.models.subnets.modules.normalization.Normalization.forward
```

````

`````
