# {py:mod}`src.models.subnets.other.grf_predictor`

```{py:module} src.models.subnets.other.grf_predictor
```

```{autodoc2-docstring} src.models.subnets.other.grf_predictor
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GatedRecurrentFillPredictor <src.models.subnets.other.grf_predictor.GatedRecurrentFillPredictor>`
  - ```{autodoc2-docstring} src.models.subnets.other.grf_predictor.GatedRecurrentFillPredictor
    :summary:
    ```
````

### API

`````{py:class} GatedRecurrentFillPredictor(input_dim=1, hidden_dim=64, num_layers=2, dropout=0.1, activation='relu', af_param=1.0, threshold=6.0, replacement_value=6.0, n_params=3, uniform_range=[0.125, 1 / 3], bidirectional=False)
:canonical: src.models.subnets.other.grf_predictor.GatedRecurrentFillPredictor

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.other.grf_predictor.GatedRecurrentFillPredictor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.other.grf_predictor.GatedRecurrentFillPredictor.__init__
```

````{py:method} forward(fill_history)
:canonical: src.models.subnets.other.grf_predictor.GatedRecurrentFillPredictor.forward

```{autodoc2-docstring} src.models.subnets.other.grf_predictor.GatedRecurrentFillPredictor.forward
```

````

````{py:method} init_hidden(batch_size, device)
:canonical: src.models.subnets.other.grf_predictor.GatedRecurrentFillPredictor.init_hidden

```{autodoc2-docstring} src.models.subnets.other.grf_predictor.GatedRecurrentFillPredictor.init_hidden
```

````

`````
