# {py:mod}`src.models.subnets.modules.dynamic_hyper_connection`

```{py:module} src.models.subnets.modules.dynamic_hyper_connection
```

```{autodoc2-docstring} src.models.subnets.modules.dynamic_hyper_connection
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DynamicHyperConnection <src.models.subnets.modules.dynamic_hyper_connection.DynamicHyperConnection>`
  - ```{autodoc2-docstring} src.models.subnets.modules.dynamic_hyper_connection.DynamicHyperConnection
    :summary:
    ```
````

### API

`````{py:class} DynamicHyperConnection(module: torch.nn.Module, embed_dim: int, n: int = 4)
:canonical: src.models.subnets.modules.dynamic_hyper_connection.DynamicHyperConnection

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.subnets.modules.dynamic_hyper_connection.DynamicHyperConnection
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.subnets.modules.dynamic_hyper_connection.DynamicHyperConnection.__init__
```

````{py:method} _initialize_identity_bias() -> None
:canonical: src.models.subnets.modules.dynamic_hyper_connection.DynamicHyperConnection._initialize_identity_bias

```{autodoc2-docstring} src.models.subnets.modules.dynamic_hyper_connection.DynamicHyperConnection._initialize_identity_bias
```

````

````{py:method} forward(H: torch.Tensor, *args: typing.Any, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.models.subnets.modules.dynamic_hyper_connection.DynamicHyperConnection.forward

```{autodoc2-docstring} src.models.subnets.modules.dynamic_hyper_connection.DynamicHyperConnection.forward
```

````

`````
