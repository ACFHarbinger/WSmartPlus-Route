# {py:mod}`src.models.modules.normalized_activation_function`

```{py:module} src.models.modules.normalized_activation_function
```

```{autodoc2-docstring} src.models.modules.normalized_activation_function
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NormalizedActivationFunction <src.models.modules.normalized_activation_function.NormalizedActivationFunction>`
  - ```{autodoc2-docstring} src.models.modules.normalized_activation_function.NormalizedActivationFunction
    :summary:
    ```
````

### API

`````{py:class} NormalizedActivationFunction(naf_name: str = 'softmax', dim: typing.Optional[int] = -1, n_classes: typing.Optional[int] = None, cutoffs: typing.Optional[typing.Sequence[int]] = None, dval: typing.Optional[float] = 4.0, bias: typing.Optional[bool] = False)
:canonical: src.models.modules.normalized_activation_function.NormalizedActivationFunction

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.modules.normalized_activation_function.NormalizedActivationFunction
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.modules.normalized_activation_function.NormalizedActivationFunction.__init__
```

````{py:method} init_parameters()
:canonical: src.models.modules.normalized_activation_function.NormalizedActivationFunction.init_parameters

```{autodoc2-docstring} src.models.modules.normalized_activation_function.NormalizedActivationFunction.init_parameters
```

````

````{py:method} forward(input, mask=None)
:canonical: src.models.modules.normalized_activation_function.NormalizedActivationFunction.forward

```{autodoc2-docstring} src.models.modules.normalized_activation_function.NormalizedActivationFunction.forward
```

````

`````
