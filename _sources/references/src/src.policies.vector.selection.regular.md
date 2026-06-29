# {py:mod}`src.policies.vector.selection.regular`

```{py:module} src.policies.vector.selection.regular
```

```{autodoc2-docstring} src.policies.vector.selection.regular
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RegularSelector <src.policies.vector.selection.regular.RegularSelector>`
  - ```{autodoc2-docstring} src.policies.vector.selection.regular.RegularSelector
    :summary:
    ```
````

### API

`````{py:class} RegularSelector(frequency: int = 3)
:canonical: src.policies.vector.selection.regular.RegularSelector

Bases: {py:obj}`src.policies.vector.selection.base.VectorizedSelector`

```{autodoc2-docstring} src.policies.vector.selection.regular.RegularSelector
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.vector.selection.regular.RegularSelector.__init__
```

````{py:method} select(fill_levels: torch.Tensor, current_day: typing.Optional[typing.Union[torch.Tensor, int]] = None, frequency: typing.Optional[int] = None, **kwargs: typing.Any) -> torch.Tensor
:canonical: src.policies.vector.selection.regular.RegularSelector.select

```{autodoc2-docstring} src.policies.vector.selection.regular.RegularSelector.select
```

````

`````
