# {py:mod}`src.models.modules.activation_function`

```{py:module} src.models.modules.activation_function
```

```{autodoc2-docstring} src.models.modules.activation_function
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ActivationFunction <src.models.modules.activation_function.ActivationFunction>`
  - ```{autodoc2-docstring} src.models.modules.activation_function.ActivationFunction
    :summary:
    ```
````

### API

`````{py:class} ActivationFunction(af_name: str = 'relu', fparam: typing.Optional[float] = None, tval: typing.Optional[float] = None, rval: typing.Optional[float] = None, n_params: typing.Optional[int] = None, urange: typing.Optional[typing.Tuple[float, float]] = None, inplace: typing.Optional[bool] = False)
:canonical: src.models.modules.activation_function.ActivationFunction

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.modules.activation_function.ActivationFunction
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.modules.activation_function.ActivationFunction.__init__
```

````{py:method} init_parameters()
:canonical: src.models.modules.activation_function.ActivationFunction.init_parameters

```{autodoc2-docstring} src.models.modules.activation_function.ActivationFunction.init_parameters
```

````

````{py:method} forward(input, mask=None)
:canonical: src.models.modules.activation_function.ActivationFunction.forward

```{autodoc2-docstring} src.models.modules.activation_function.ActivationFunction.forward
```

````

`````
