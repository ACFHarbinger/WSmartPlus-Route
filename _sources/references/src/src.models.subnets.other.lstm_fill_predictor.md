# {py:mod}`src.models.subnets.other.lstm_fill_predictor`

```{py:module} src.models.subnets.other.lstm_fill_predictor
```

```{autodoc2-docstring} src.models.subnets.other.lstm_fill_predictor
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LongShortTermMemoryFillPredictor <src.models.subnets.other.lstm_fill_predictor.LongShortTermMemoryFillPredictor>`
  - ```{autodoc2-docstring} src.models.subnets.other.lstm_fill_predictor.LongShortTermMemoryFillPredictor
    :summary:
    ```
````

### API

`````{py:class} LongShortTermMemoryFillPredictor(input_dim=1, hidden_dim=64, num_layers=2, dropout=0.1, activation='relu', af_param=1.0, threshold=6.0, replacement_value=6.0, n_params=3, uniform_range=None, bidirectional=False)
:canonical: src.models.subnets.other.lstm_fill_predictor.LongShortTermMemoryFillPredictor

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.other.lstm_fill_predictor.LongShortTermMemoryFillPredictor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.other.lstm_fill_predictor.LongShortTermMemoryFillPredictor.__init__
```

````{py:method} forward(fill_history)
:canonical: src.models.subnets.other.lstm_fill_predictor.LongShortTermMemoryFillPredictor.forward

```{autodoc2-docstring} src.models.subnets.other.lstm_fill_predictor.LongShortTermMemoryFillPredictor.forward
```

````

````{py:method} get_embedding(fill_history)
:canonical: src.models.subnets.other.lstm_fill_predictor.LongShortTermMemoryFillPredictor.get_embedding

```{autodoc2-docstring} src.models.subnets.other.lstm_fill_predictor.LongShortTermMemoryFillPredictor.get_embedding
```

````

`````
