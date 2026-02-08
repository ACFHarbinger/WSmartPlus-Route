# {py:mod}`src.models.hrl_manager.temporal_encoder`

```{py:module} src.models.hrl_manager.temporal_encoder
```

```{autodoc2-docstring} src.models.hrl_manager.temporal_encoder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TemporalEncoder <src.models.hrl_manager.temporal_encoder.TemporalEncoder>`
  - ```{autodoc2-docstring} src.models.hrl_manager.temporal_encoder.TemporalEncoder
    :summary:
    ```
````

### API

`````{py:class} TemporalEncoder(hidden_dim: int = 64, rnn_type: str = 'lstm')
:canonical: src.models.hrl_manager.temporal_encoder.TemporalEncoder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.hrl_manager.temporal_encoder.TemporalEncoder
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.hrl_manager.temporal_encoder.TemporalEncoder.__init__
```

````{py:method} forward(dynamic: torch.Tensor) -> torch.Tensor
:canonical: src.models.hrl_manager.temporal_encoder.TemporalEncoder.forward

```{autodoc2-docstring} src.models.hrl_manager.temporal_encoder.TemporalEncoder.forward
```

````

`````
