# {py:mod}`src.models.policies.selection.base`

```{py:module} src.models.policies.selection.base
```

```{autodoc2-docstring} src.models.policies.selection.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VectorizedSelector <src.models.policies.selection.base.VectorizedSelector>`
  - ```{autodoc2-docstring} src.models.policies.selection.base.VectorizedSelector
    :summary:
    ```
````

### API

`````{py:class} VectorizedSelector
:canonical: src.models.policies.selection.base.VectorizedSelector

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.models.policies.selection.base.VectorizedSelector
```

````{py:method} select(fill_levels: torch.Tensor, **kwargs) -> torch.Tensor
:canonical: src.models.policies.selection.base.VectorizedSelector.select
:abstractmethod:

```{autodoc2-docstring} src.models.policies.selection.base.VectorizedSelector.select
```

````

`````
