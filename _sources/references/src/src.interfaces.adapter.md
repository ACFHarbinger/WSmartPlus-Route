# {py:mod}`src.interfaces.adapter`

```{py:module} src.interfaces.adapter
```

```{autodoc2-docstring} src.interfaces.adapter
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IPolicyAdapter <src.interfaces.adapter.IPolicyAdapter>`
  - ```{autodoc2-docstring} src.interfaces.adapter.IPolicyAdapter
    :summary:
    ```
````

### API

`````{py:class} IPolicyAdapter
:canonical: src.interfaces.adapter.IPolicyAdapter

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.interfaces.adapter.IPolicyAdapter
```

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, typing.Any]
:canonical: src.interfaces.adapter.IPolicyAdapter.execute
:abstractmethod:

```{autodoc2-docstring} src.interfaces.adapter.IPolicyAdapter.execute
```

````

`````
