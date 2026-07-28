# {py:mod}`src.interfaces.tensor_dict_like`

```{py:module} src.interfaces.tensor_dict_like
```

```{autodoc2-docstring} src.interfaces.tensor_dict_like
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ITensorDictLike <src.interfaces.tensor_dict_like.ITensorDictLike>`
  - ```{autodoc2-docstring} src.interfaces.tensor_dict_like.ITensorDictLike
    :summary:
    ```
````

### API

`````{py:class} ITensorDictLike
:canonical: src.interfaces.tensor_dict_like.ITensorDictLike

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} src.interfaces.tensor_dict_like.ITensorDictLike
```

````{py:method} get(key: str, default: typing.Optional[typing.Any] = None) -> typing.Any
:canonical: src.interfaces.tensor_dict_like.ITensorDictLike.get

```{autodoc2-docstring} src.interfaces.tensor_dict_like.ITensorDictLike.get
```

````

````{py:method} set(key: str, value: torch.Tensor) -> None
:canonical: src.interfaces.tensor_dict_like.ITensorDictLike.set

```{autodoc2-docstring} src.interfaces.tensor_dict_like.ITensorDictLike.set
```

````

````{py:method} __getitem__(key: str) -> torch.Tensor
:canonical: src.interfaces.tensor_dict_like.ITensorDictLike.__getitem__

```{autodoc2-docstring} src.interfaces.tensor_dict_like.ITensorDictLike.__getitem__
```

````

````{py:method} __contains__(key: str) -> bool
:canonical: src.interfaces.tensor_dict_like.ITensorDictLike.__contains__

```{autodoc2-docstring} src.interfaces.tensor_dict_like.ITensorDictLike.__contains__
```

````

````{py:method} values() -> typing.Any
:canonical: src.interfaces.tensor_dict_like.ITensorDictLike.values

```{autodoc2-docstring} src.interfaces.tensor_dict_like.ITensorDictLike.values
```

````

````{py:method} keys() -> typing.Any
:canonical: src.interfaces.tensor_dict_like.ITensorDictLike.keys

```{autodoc2-docstring} src.interfaces.tensor_dict_like.ITensorDictLike.keys
```

````

````{py:method} items() -> typing.Any
:canonical: src.interfaces.tensor_dict_like.ITensorDictLike.items

```{autodoc2-docstring} src.interfaces.tensor_dict_like.ITensorDictLike.items
```

````

````{py:property} batch_size
:canonical: src.interfaces.tensor_dict_like.ITensorDictLike.batch_size
:type: typing.Tuple[int, ...]

```{autodoc2-docstring} src.interfaces.tensor_dict_like.ITensorDictLike.batch_size
```

````

````{py:property} device
:canonical: src.interfaces.tensor_dict_like.ITensorDictLike.device
:type: torch.device

```{autodoc2-docstring} src.interfaces.tensor_dict_like.ITensorDictLike.device
```

````

`````
