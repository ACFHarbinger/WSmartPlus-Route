# {py:mod}`src.models.meta.weight_adjustment_rnn.model`

```{py:module} src.models.meta.weight_adjustment_rnn.model
```

```{autodoc2-docstring} src.models.meta.weight_adjustment_rnn.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`WeightAdjustmentRNN <src.models.meta.weight_adjustment_rnn.model.WeightAdjustmentRNN>`
  - ```{autodoc2-docstring} src.models.meta.weight_adjustment_rnn.model.WeightAdjustmentRNN
    :summary:
    ```
````

### API

`````{py:class} WeightAdjustmentRNN(input_size: int, hidden_size: int, output_size: int, num_layers: int = 1)
:canonical: src.models.meta.weight_adjustment_rnn.model.WeightAdjustmentRNN

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.meta.weight_adjustment_rnn.model.WeightAdjustmentRNN
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.meta.weight_adjustment_rnn.model.WeightAdjustmentRNN.__init__
```

````{py:method} init_weights() -> None
:canonical: src.models.meta.weight_adjustment_rnn.model.WeightAdjustmentRNN.init_weights

```{autodoc2-docstring} src.models.meta.weight_adjustment_rnn.model.WeightAdjustmentRNN.init_weights
```

````

````{py:method} forward(x: torch.Tensor, hidden: typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]] = None) -> typing.Tuple[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]
:canonical: src.models.meta.weight_adjustment_rnn.model.WeightAdjustmentRNN.forward

```{autodoc2-docstring} src.models.meta.weight_adjustment_rnn.model.WeightAdjustmentRNN.forward
```

````

`````
