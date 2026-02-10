# {py:mod}`src.interfaces.traversable`

```{py:module} src.interfaces.traversable
```

```{autodoc2-docstring} src.interfaces.traversable
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ITraversable <src.interfaces.traversable.ITraversable>`
  - ```{autodoc2-docstring} src.interfaces.traversable.ITraversable
    :summary:
    ```
````

### API

`````{py:class} ITraversable
:canonical: src.interfaces.traversable.ITraversable

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} src.interfaces.traversable.ITraversable
```

````{py:method} __getitem__(key: str) -> typing.Any
:canonical: src.interfaces.traversable.ITraversable.__getitem__

```{autodoc2-docstring} src.interfaces.traversable.ITraversable.__getitem__
```

````

````{py:method} __contains__(key: str) -> bool
:canonical: src.interfaces.traversable.ITraversable.__contains__

```{autodoc2-docstring} src.interfaces.traversable.ITraversable.__contains__
```

````

````{py:method} keys() -> typing.Any
:canonical: src.interfaces.traversable.ITraversable.keys

```{autodoc2-docstring} src.interfaces.traversable.ITraversable.keys
```

````

````{py:method} items() -> typing.Any
:canonical: src.interfaces.traversable.ITraversable.items

```{autodoc2-docstring} src.interfaces.traversable.ITraversable.items
```

````

````{py:method} values() -> typing.Any
:canonical: src.interfaces.traversable.ITraversable.values

```{autodoc2-docstring} src.interfaces.traversable.ITraversable.values
```

````

````{py:method} get(key: str, default: typing.Optional[typing.Any] = None) -> typing.Any
:canonical: src.interfaces.traversable.ITraversable.get

```{autodoc2-docstring} src.interfaces.traversable.ITraversable.get
```

````

`````
