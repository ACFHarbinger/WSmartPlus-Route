# {py:mod}`src.models.modules.connections`

```{py:module} src.models.modules.connections
```

```{autodoc2-docstring} src.models.modules.connections
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Connections <src.models.modules.connections.Connections>`
  - ```{autodoc2-docstring} src.models.modules.connections.Connections
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_connection_module <src.models.modules.connections.get_connection_module>`
  - ```{autodoc2-docstring} src.models.modules.connections.get_connection_module
    :summary:
    ```
````

### API

````{py:class} Connections()
:canonical: src.models.modules.connections.Connections

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} src.models.modules.connections.Connections
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.modules.connections.Connections.__init__
```

````

````{py:function} get_connection_module(module, embed_dim, connection_type='skip', **kwargs)
:canonical: src.models.modules.connections.get_connection_module

```{autodoc2-docstring} src.models.modules.connections.get_connection_module
```
````
