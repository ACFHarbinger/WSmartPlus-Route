# {py:mod}`src.models.n2s.encoder`

```{py:module} src.models.n2s.encoder
```

```{autodoc2-docstring} src.models.n2s.encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`N2SEncoder <src.models.n2s.encoder.N2SEncoder>`
  - ```{autodoc2-docstring} src.models.n2s.encoder.N2SEncoder
    :summary:
    ```
````

### API

`````{py:class} N2SEncoder(embed_dim: int = 128, num_heads: int = 8, k_neighbors: int = 20, **kwargs)
:canonical: src.models.n2s.encoder.N2SEncoder

Bases: {py:obj}`src.models.common.improvement_encoder.ImprovementEncoder`

```{autodoc2-docstring} src.models.n2s.encoder.N2SEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.n2s.encoder.N2SEncoder.__init__
```

````{py:method} _get_neighborhood_mask(td: tensordict.TensorDict) -> torch.Tensor
:canonical: src.models.n2s.encoder.N2SEncoder._get_neighborhood_mask

```{autodoc2-docstring} src.models.n2s.encoder.N2SEncoder._get_neighborhood_mask
```

````

````{py:method} forward(td: tensordict.TensorDict, **kwargs) -> torch.Tensor
:canonical: src.models.n2s.encoder.N2SEncoder.forward

```{autodoc2-docstring} src.models.n2s.encoder.N2SEncoder.forward
```

````

`````
