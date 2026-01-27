# {py:mod}`src.models.modules.hyper_connection`

```{py:module} src.models.modules.hyper_connection
```

```{autodoc2-docstring} src.models.modules.hyper_connection
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StaticHyperConnection <src.models.modules.hyper_connection.StaticHyperConnection>`
  - ```{autodoc2-docstring} src.models.modules.hyper_connection.StaticHyperConnection
    :summary:
    ```
* - {py:obj}`DynamicHyperConnection <src.models.modules.hyper_connection.DynamicHyperConnection>`
  - ```{autodoc2-docstring} src.models.modules.hyper_connection.DynamicHyperConnection
    :summary:
    ```
````

### API

`````{py:class} StaticHyperConnection(module: torch.nn.Module, hyper_dim: int, expansion_rate: int = 4)
:canonical: src.models.modules.hyper_connection.StaticHyperConnection

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.modules.hyper_connection.StaticHyperConnection
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.modules.hyper_connection.StaticHyperConnection.__init__
```

````{py:method} forward(H, *args, **kwargs)
:canonical: src.models.modules.hyper_connection.StaticHyperConnection.forward

```{autodoc2-docstring} src.models.modules.hyper_connection.StaticHyperConnection.forward
```

````

`````

`````{py:class} DynamicHyperConnection(module, embed_dim, n=4)
:canonical: src.models.modules.hyper_connection.DynamicHyperConnection

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.modules.hyper_connection.DynamicHyperConnection
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.modules.hyper_connection.DynamicHyperConnection.__init__
```

````{py:method} _initialize_identity_bias()
:canonical: src.models.modules.hyper_connection.DynamicHyperConnection._initialize_identity_bias

```{autodoc2-docstring} src.models.modules.hyper_connection.DynamicHyperConnection._initialize_identity_bias
```

````

````{py:method} forward(H, *args, **kwargs)
:canonical: src.models.modules.hyper_connection.DynamicHyperConnection.forward

```{autodoc2-docstring} src.models.modules.hyper_connection.DynamicHyperConnection.forward
```

````

`````
