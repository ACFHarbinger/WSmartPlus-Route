# {py:mod}`src.interfaces.model`

```{py:module} src.interfaces.model
```

```{autodoc2-docstring} src.interfaces.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IModel <src.interfaces.model.IModel>`
  - ```{autodoc2-docstring} src.interfaces.model.IModel
    :summary:
    ```
````

### API

`````{py:class} IModel
:canonical: src.interfaces.model.IModel

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} src.interfaces.model.IModel
```

````{py:method} forward(td: tensordict.TensorDict, **kwargs: typing.Any) -> typing.Any
:canonical: src.interfaces.model.IModel.forward

```{autodoc2-docstring} src.interfaces.model.IModel.forward
```

````

````{py:method} to(device: torch.device) -> src.interfaces.model.IModel
:canonical: src.interfaces.model.IModel.to

```{autodoc2-docstring} src.interfaces.model.IModel.to
```

````

````{py:method} eval() -> src.interfaces.model.IModel
:canonical: src.interfaces.model.IModel.eval

```{autodoc2-docstring} src.interfaces.model.IModel.eval
```

````

````{py:method} train() -> src.interfaces.model.IModel
:canonical: src.interfaces.model.IModel.train

```{autodoc2-docstring} src.interfaces.model.IModel.train
```

````

`````
