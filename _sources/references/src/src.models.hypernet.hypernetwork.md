# {py:mod}`src.models.hypernet.hypernetwork`

```{py:module} src.models.hypernet.hypernetwork
```

```{autodoc2-docstring} src.models.hypernet.hypernetwork
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HyperNetwork <src.models.hypernet.hypernetwork.HyperNetwork>`
  - ```{autodoc2-docstring} src.models.hypernet.hypernetwork.HyperNetwork
    :summary:
    ```
````

### API

`````{py:class} HyperNetwork(input_dim, output_dim, n_days=365, embed_dim=16, hidden_dim=64, normalization='layer', activation='relu', learn_affine=True, bias=True)
:canonical: src.models.hypernet.hypernetwork.HyperNetwork

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.hypernet.hypernetwork.HyperNetwork
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.hypernet.hypernetwork.HyperNetwork.__init__
```

````{py:method} init_weights()
:canonical: src.models.hypernet.hypernetwork.HyperNetwork.init_weights

```{autodoc2-docstring} src.models.hypernet.hypernetwork.HyperNetwork.init_weights
```

````

````{py:method} forward(metrics, day)
:canonical: src.models.hypernet.hypernetwork.HyperNetwork.forward

```{autodoc2-docstring} src.models.hypernet.hypernetwork.HyperNetwork.forward
```

````

`````
