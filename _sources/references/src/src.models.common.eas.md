# {py:mod}`src.models.common.eas`

```{py:module} src.models.common.eas
```

```{autodoc2-docstring} src.models.common.eas
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EAS <src.models.common.eas.EAS>`
  - ```{autodoc2-docstring} src.models.common.eas.EAS
    :summary:
    ```
````

### API

`````{py:class} EAS(model: torch.nn.Module, optimizer_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, n_search_steps: int = 20, search_param_names: typing.Optional[typing.List[str]] = None, **kwargs: typing.Any)
:canonical: src.models.common.eas.EAS

Bases: {py:obj}`src.models.common.transductive_base.TransductiveModel`

```{autodoc2-docstring} src.models.common.eas.EAS
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.eas.EAS.__init__
```

````{py:method} _get_search_params() -> typing.Any
:canonical: src.models.common.eas.EAS._get_search_params

```{autodoc2-docstring} src.models.common.eas.EAS._get_search_params
```

````

`````
