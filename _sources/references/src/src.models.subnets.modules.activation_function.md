# {py:mod}`src.models.subnets.modules.activation_function`

```{py:module} src.models.subnets.modules.activation_function
```

```{autodoc2-docstring} src.models.subnets.modules.activation_function
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ActivationFunction <src.models.subnets.modules.activation_function.ActivationFunction>`
  - ```{autodoc2-docstring} src.models.subnets.modules.activation_function.ActivationFunction
    :summary:
    ```
````

### API

`````{py:class} ActivationFunction(af_name: typing.Optional[str] = None, fparam: typing.Optional[float] = None, tval: typing.Optional[float] = None, rval: typing.Optional[float] = None, n_params: typing.Optional[int] = None, urange: typing.Optional[typing.Tuple[float, float]] = None, inplace: typing.Optional[bool] = False, activation_config: typing.Optional[logic.src.configs.models.activation_function.ActivationConfig] = None)
:canonical: src.models.subnets.modules.activation_function.ActivationFunction

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.modules.activation_function.ActivationFunction
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.activation_function.ActivationFunction.__init__
```

````{py:method} _get_activation_module(af_name: str, fparam: typing.Optional[float], tval: typing.Optional[float], rval: typing.Optional[float], n_params: typing.Optional[int], urange: typing.Optional[typing.Tuple[float, float]], inplace: typing.Optional[bool]) -> torch.nn.Module
:canonical: src.models.subnets.modules.activation_function.ActivationFunction._get_activation_module

```{autodoc2-docstring} src.models.subnets.modules.activation_function.ActivationFunction._get_activation_module
```

````

````{py:method} init_parameters()
:canonical: src.models.subnets.modules.activation_function.ActivationFunction.init_parameters

```{autodoc2-docstring} src.models.subnets.modules.activation_function.ActivationFunction.init_parameters
```

````

````{py:method} forward(input, mask=None)
:canonical: src.models.subnets.modules.activation_function.ActivationFunction.forward

```{autodoc2-docstring} src.models.subnets.modules.activation_function.ActivationFunction.forward
```

````

`````
