# {py:mod}`src.models.meta_rnn`

```{py:module} src.models.meta_rnn
```

```{autodoc2-docstring} src.models.meta_rnn
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`WeightAdjustmentRNN <src.models.meta_rnn.WeightAdjustmentRNN>`
  - ```{autodoc2-docstring} src.models.meta_rnn.WeightAdjustmentRNN
    :summary:
    ```
````

### API

`````{py:class} WeightAdjustmentRNN(input_size, hidden_size, output_size, num_layers=1)
:canonical: src.models.meta_rnn.WeightAdjustmentRNN

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.meta_rnn.WeightAdjustmentRNN
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.meta_rnn.WeightAdjustmentRNN.__init__
```

````{py:method} init_weights()
:canonical: src.models.meta_rnn.WeightAdjustmentRNN.init_weights

```{autodoc2-docstring} src.models.meta_rnn.WeightAdjustmentRNN.init_weights
```

````

````{py:method} forward(x, hidden=None)
:canonical: src.models.meta_rnn.WeightAdjustmentRNN.forward

```{autodoc2-docstring} src.models.meta_rnn.WeightAdjustmentRNN.forward
```

````

`````
