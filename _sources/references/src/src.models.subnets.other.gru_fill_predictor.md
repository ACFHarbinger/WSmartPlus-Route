# {py:mod}`src.models.subnets.other.gru_fill_predictor`

```{py:module} src.models.subnets.other.gru_fill_predictor
```

```{autodoc2-docstring} src.models.subnets.other.gru_fill_predictor
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GatedRecurrentUnitFillPredictor <src.models.subnets.other.gru_fill_predictor.GatedRecurrentUnitFillPredictor>`
  - ```{autodoc2-docstring} src.models.subnets.other.gru_fill_predictor.GatedRecurrentUnitFillPredictor
    :summary:
    ```
````

### API

`````{py:class} GatedRecurrentUnitFillPredictor(input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1, activation: str = 'relu', af_param: float = 1.0, threshold: float = 6.0, replacement_value: float = 6.0, n_params: int = 3, uniform_range: typing.Optional[typing.List[float]] = None, bidirectional: bool = False)
:canonical: src.models.subnets.other.gru_fill_predictor.GatedRecurrentUnitFillPredictor

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.other.gru_fill_predictor.GatedRecurrentUnitFillPredictor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.other.gru_fill_predictor.GatedRecurrentUnitFillPredictor.__init__
```

````{py:method} forward(fill_history: torch.Tensor) -> torch.Tensor
:canonical: src.models.subnets.other.gru_fill_predictor.GatedRecurrentUnitFillPredictor.forward

```{autodoc2-docstring} src.models.subnets.other.gru_fill_predictor.GatedRecurrentUnitFillPredictor.forward
```

````

````{py:method} init_hidden(batch_size: int, device: torch.device) -> torch.Tensor
:canonical: src.models.subnets.other.gru_fill_predictor.GatedRecurrentUnitFillPredictor.init_hidden

```{autodoc2-docstring} src.models.subnets.other.gru_fill_predictor.GatedRecurrentUnitFillPredictor.init_hidden
```

````

`````
