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

* - {py:obj}`DACAttLayer <src.models.core.dact.encoder.DACAttLayer>`
  - ```{autodoc2-docstring} src.models.core.dact.encoder.DACAttLayer
    :summary:
    ```
* - {py:obj}`DACTEncoder <src.models.core.dact.encoder.DACTEncoder>`
  - ```{autodoc2-docstring} src.models.core.dact.encoder.DACTEncoder
    :summary:
    ```
````

### API

`````{py:class} DACAttLayer(embed_dim: int, num_heads: int = 4)
:canonical: src.models.core.dact.encoder.DACAttLayer

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.core.dact.encoder.DACAttLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.dact.encoder.DACAttLayer.__init__
```

````{py:method} forward(h: torch.Tensor, g: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.core.dact.encoder.DACAttLayer.forward

```{autodoc2-docstring} src.models.core.dact.encoder.DACAttLayer.forward
```

````

`````

`````{py:class} DACTEncoder(embed_dim: int = 128, num_layers: int = 3, num_heads: int = 4, pos_type: str = 'CPE', **kwargs: typing.Any)
:canonical: src.models.core.dact.encoder.DACTEncoder

Bases: {py:obj}`logic.src.models.common.improvement.encoder.ImprovementEncoder`

```{autodoc2-docstring} src.models.core.dact.encoder.DACTEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.core.dact.encoder.DACTEncoder.__init__
```

````{py:method} _compute_cyclic_positions(solution: torch.Tensor, n_nodes: int) -> torch.Tensor
:canonical: src.models.core.dact.encoder.DACTEncoder._compute_cyclic_positions

```{autodoc2-docstring} src.models.core.dact.encoder.DACTEncoder._compute_cyclic_positions
```

````

````{py:method} forward(td: tensordict.TensorDict, **kwargs: typing.Any) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: src.models.core.dact.encoder.DACTEncoder.forward

```{autodoc2-docstring} src.models.core.dact.encoder.DACTEncoder.forward
```

````

`````
