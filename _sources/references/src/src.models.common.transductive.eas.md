# {py:mod}`src.models.common.transductive.eas`

```{py:module} src.models.common.transductive.eas
```

```{autodoc2-docstring} src.models.common.transductive.eas
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EAS <src.models.common.transductive.eas.EAS>`
  - ```{autodoc2-docstring} src.models.common.transductive.eas.EAS
    :summary:
    ```
````

### API

`````{py:class} EAS(model: torch.nn.Module, optimizer_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, n_search_steps: int = 20, search_param_names: typing.Optional[typing.List[str]] = None, **kwargs: typing.Any)
:canonical: src.models.common.transductive.eas.EAS

Bases: {py:obj}`src.models.common.transductive.base.TransductiveModel`

```{autodoc2-docstring} src.models.common.transductive.eas.EAS
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.models.common.transductive.eas.EAS.__init__
```

````{py:method} _get_search_params() -> typing.Any
:canonical: src.models.common.transductive.eas.EAS._get_search_params

```{autodoc2-docstring} src.models.common.transductive.eas.EAS._get_search_params
```

````

`````
