# {py:mod}`src.interfaces.bin_container`

```{py:module} src.interfaces.bin_container
```

```{autodoc2-docstring} src.interfaces.bin_container
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IBinContainer <src.interfaces.bin_container.IBinContainer>`
  - ```{autodoc2-docstring} src.interfaces.bin_container.IBinContainer
    :summary:
    ```
````

### API

`````{py:class} IBinContainer
:canonical: src.interfaces.bin_container.IBinContainer

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} src.interfaces.bin_container.IBinContainer
```

````{py:property} fill_levels
:canonical: src.interfaces.bin_container.IBinContainer.fill_levels
:type: torch.Tensor

```{autodoc2-docstring} src.interfaces.bin_container.IBinContainer.fill_levels
```

````

````{py:property} demands
:canonical: src.interfaces.bin_container.IBinContainer.demands
:type: torch.Tensor

```{autodoc2-docstring} src.interfaces.bin_container.IBinContainer.demands
```

````

````{py:method} update_fill_levels(visited: torch.Tensor) -> None
:canonical: src.interfaces.bin_container.IBinContainer.update_fill_levels

```{autodoc2-docstring} src.interfaces.bin_container.IBinContainer.update_fill_levels
```

````

````{py:method} get(key: str, default: typing.Optional[typing.Any] = None) -> typing.Any
:canonical: src.interfaces.bin_container.IBinContainer.get

```{autodoc2-docstring} src.interfaces.bin_container.IBinContainer.get
```

````

`````
